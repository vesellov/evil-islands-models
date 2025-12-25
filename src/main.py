import os
import sys
import math


_Debug = True


from kivy.config import Config
# Config.set('graphics', 'window_state', 'maximized')

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix  # @UnresolvedImport
from kivy.graphics.opengl import glGetError, glEnable, glDisable, GL_DEPTH_TEST  # @UnresolvedImport
from kivy.graphics.instructions import InstructionGroup  # @UnresolvedImport
from kivy.graphics.context_instructions import Transform  # @UnresolvedImport
from kivy.graphics import (
    RenderContext, Callback, BindTexture,
    ChangeState, PushState, PopState,
    PushMatrix, PopMatrix, Scale,
    Color, Translate, Rotate, Mesh, UpdateNormalMatrix,
)

import dat
import mth
import res


vertex_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_normal;
attribute vec2  v_tex_coord;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec2 tex_coord0;
varying vec4 normal_vec;
varying vec4 vertex_pos;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos, 1.0);
    vertex_pos = pos;
    normal_vec = vec4(v_normal,0.0);
    gl_Position = projection_mat * pos;
    tex_coord0 = v_tex_coord;
}
"""

fragment_shader_src = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 normal_vec;
varying vec4 vertex_pos;
varying vec2 tex_coord0;

uniform sampler2D texture_id;
uniform mat4 normal_mat;
uniform vec4 line_color;

void main (void) {
    gl_FragColor = texture2D(texture_id, tex_coord0) * line_color;
}
"""


def ignore_undertouch(func):
    def wrap(self, touch):
        glst = touch.grab_list
        if len(glst) == 0 or (self is glst[ 0 ]()):
            return func(self, touch)
    return wrap


class Renderer(Widget):

    SCALE_FACTOR = 0.05
    MAX_SCALE = 10.0
    MIN_SCALE = 0.1
    ROTATE_SPEED = 1.

    def __init__(self, **kwargs):
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        self.coord_x = 0
        self.coord_y = 0
        self.coord_z = 3
        self._touches = []
        self.canvas = RenderContext(compute_normal_mat=True)
        self.canvas.shader.fs = fragment_shader_src
        self.canvas.shader.vs = vertex_shader_src
        self.container = None
        self.meshes_onstage = set()
        self.scene = dat.SceneData()
        self.models_data = {}
        self.test_id = 1
        self.prepare_test_unit(test=self.test_id)
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        self.canvas['texture_id'] = 1
        Clock.schedule_interval(self.update_glsl, 1 / 15)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        for name in self.scene.units.keys():
            self.add_unit(name)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'escape':
            App.get_running_app().stop()
        elif keycode[1] == 'z':
            for u in self.scene.units.values():
                if not u.onstage:
                    continue
                current_animation_ind = u.animations_loaded.index(u.animation_playing)
                current_animation_ind += 1
                if current_animation_ind >= len(u.animations_loaded):
                    current_animation_ind = 0
                u.animation_playing = u.animations_loaded[current_animation_ind]
                u.animation_frame = 0
                if _Debug:
                    print(f'playing animation {u.animation_playing} for ({u.name})')
                break
        elif keycode[1] == 'x':
            for u in self.scene.units.values():
                if not u.onstage:
                    continue
                current_animation_ind = u.animations_loaded.index(u.animation_playing)
                current_animation_ind -= 1
                if current_animation_ind < 0:
                    current_animation_ind = len(u.animations_loaded) - 1
                u.animation_playing = u.animations_loaded[current_animation_ind]
                u.animation_frame = 0
                if _Debug:
                    print(f'playing animation {u.animation_playing} for ({u.name})')
                break
        elif keycode[1] == 'c':
            units_onstage = []
            for unit in self.scene.units.values():
                if unit.onstage:
                    units_onstage.append(unit.name)
            for name in units_onstage:
                self.remove_unit(name)
            self.test_id += 1
            if self.test_id > 3:
                self.test_id = 1
            name = self.prepare_test_unit(test=self.test_id)
            self.add_unit(name)
        elif keycode[1] == 'v':
            units_onstage = []
            for unit in self.scene.units.values():
                if unit.onstage:
                    units_onstage.append(unit.name)
            for name in units_onstage:
                self.remove_unit(name)
            self.test_id -= 1
            if self.test_id == 0:
                self.test_id = 3
            name = self.prepare_test_unit(test=self.test_id)
            self.add_unit(name)
        return True

    @ignore_undertouch
    def on_touch_down(self, touch):
        touch.grab(self)
        self._touches.append(touch)
        if 'button' in touch.profile and touch.button in ('scrollup', 'scrolldown'):
            if touch.button == "scrolldown":
                scale = self.SCALE_FACTOR
            if touch.button == "scrollup":
                scale = -self.SCALE_FACTOR
            xyz = self.global_scale.xyz
            scale = xyz[0] + scale
            if scale < self.MAX_SCALE and scale > self.MIN_SCALE:
                self.global_scale.xyz = (scale, scale, scale)

    @ignore_undertouch
    def on_touch_up(self, touch):
        touch.ungrab(self)
        if touch in self._touches:
            self._touches.remove(touch)

    def define_rotate_angle(self, touch):
        x_angle = (touch.dx / self.width) * 360.0 * self.ROTATE_SPEED
        y_angle = -1 * (touch.dy / self.height) * 360.0 * self.ROTATE_SPEED
        return x_angle, y_angle

    @ignore_undertouch
    def on_touch_move(self, touch):
        if touch in self._touches and touch.grab_current == self:
            if len(self._touches) == 1:
                ax, ay = self.define_rotate_angle(touch)
                self.global_rotate_y.angle -= ax
                self.global_rotate_x.angle -= ay

    def add_mesh(self, name):
        # NOT TO BE USED
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.meshes:
            raise Exception(f'Mesh {name} does not exist')
        mesh = self.scene.meshes.get(name)
        self.container.add(PushMatrix(group=mesh.name))
        self.container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=mesh.name))
        self.container.add(Mesh(
            vertices=mesh.vertices,
            indices=mesh.indices,
            fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
            mode='triangles',
            group=mesh.name,
        ))
        self.container.add(PopMatrix(group=mesh.name))
        self.meshes_onstage.add(name)
        mesh.onstage = True
        if _Debug:
            print(f'added mesh <{name}> on scene with {len(mesh.vertices)} vertices and {len(mesh.indices)} indices')

    def remove_mesh(self, name):
        # NOT TO BE USED
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.meshes:
            raise Exception(f'Mesh {name} does not exist')
        mesh = self.scene.meshes[name]
        self.container.remove_group(name)
        self.meshes_onstage.remove(name)
        mesh.onstage = False
        if _Debug:
            print(f'removed mesh <{name}> from scene')

    def add_unit(self, name):
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.units:
            raise Exception(f'Unit {name} does not exist')
        unit = self.scene.units[name]
        
        def _visitor(part_name, parent_part_name):
            # if _Debug:
            #     print(f'    pushed matrix {part_name}')
            mesh = unit.meshes.get(part_name)
            self.container.add(PushMatrix(group=mesh.name))
            mesh.part_translate = Transform(group=mesh.name)
            self.container.add(mesh.part_translate)
            self.container.add(PushMatrix(group=mesh.name))
            mesh.part_rotate = Transform(group=mesh.name)
            self.container.add(mesh.part_rotate)
            # TODO: check if we can pass texture data directly to Mesh instruction as a parameter
            self.container.add(BindTexture(source=mesh.material['map_Kd'], index=1, group=mesh.name))
            self.container.add(Mesh(
                vertices=mesh.vertices,
                indices=mesh.indices,
                fmt=[(b'v_pos', 3, 'float'), (b'v_normal', 3, 'float'), (b'v_tex_coord', 2, 'float')],
                mode='triangles',
                group=mesh.name,
                # texture=<already loaded Texture>,
            ))
            self.container.add(PopMatrix(group=mesh.name))  # part_rotate
            self.container.add(PopMatrix(group=mesh.name))  # part_translate
            self.meshes_onstage.add(mesh.name)
            mesh.onstage = True
            # if _Debug:
            #     print(f'    popped matrix {part_name}')

        self.container.add(PushMatrix(group=unit.name))  # unit
        unit.walk_parts_ordered(_visitor)
        self.container.add(PopMatrix(group=unit.name))  # unit
        unit.onstage = True
        if _Debug:
            print(f'added unit ({unit.name}) on the stage')

    def remove_unit(self, name):
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.units:
            raise Exception(f'Unit {name} does not exist')
        unit = self.scene.units[name]
        unit.onstage = False
        for mesh in unit.meshes.values():
            mesh.onstage = False
            self.container.remove_group(mesh.name)
            mesh.part_rotate = None
            mesh.part_translate = None
            self.meshes_onstage.remove(mesh.name)
        self.container.remove_group(unit.name)
        if _Debug:
            print(f'removed mesh unit ({unit.name}) from scene')

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def gl_error(self, text='', kill=True):
        err = glGetError()
        if not err:
            return 
        while err:
            if _Debug:
                print('## GL ## = ' + text + 'OPENGL Error Code = ' + str(err))
            err = glGetError()
        if kill == True:
            sys.exit(0)

    def update_glsl(self, delta):
        asp = self.width / float(self.height)
        self.gl_error('step 1')
        self.canvas['texture_id'] = 1
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(
            0, 0, -5,
            0, 0, 0,
            0, 1, 0,
        )
        self.canvas['diffuse_light'] = (1.0, 1.0, 1.0)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.gl_error('step 2')
        # TODO: maintain separate list of active animations for all units
        # then it is not required to loop all units
        for unit in self.scene.units.values():
            if not unit.onstage:
                continue
            if not unit.animation_playing:
                continue
            parts_animations = unit.animations[unit.animation_playing]
            root_part_name = unit.parts[0]
            root_part_animation_info = parts_animations.get(root_part_name)
            if unit.animation_frame >= len(root_part_animation_info['absolute_rotation_frames']):
                if _Debug:
                    print(f'restarting unit ({unit.name}) animation {unit.animation_playing} after frame {unit.animation_frame}')
                unit.animation_frame = 0
            for part_name in unit.parts:
                if part_name not in parts_animations:
                    continue
                part_animation_info = parts_animations.get(part_name)
                q = part_animation_info['absolute_rotation_frames'][unit.animation_frame]
                t = part_animation_info['absolute_translation_frames'][unit.animation_frame]
                mesh = unit.meshes[part_name]
                translate_mat = Matrix()
                translate_mat.translate(t[0], t[1], t[2])
                mesh.part_translate.matrix = translate_mat
                rotate_mat = Matrix()
                rotate_mat.set(array=mth.quaternion_to_matrix(q[0], q[1], q[2], q[3]))
                mesh.part_rotate.matrix = rotate_mat.inverse()
            unit.animation_frame += 1

    def setup_scene(self):
        PushMatrix()
        self.global_translate = Translate(0, 0, 0)
        self.global_rotate_x = Rotate(0, 1, 0, 0)
        self.global_rotate_y = Rotate(0, 0, 1, 0)
        self.global_scale = Scale(0.5)
        sz = 1
        PushState()
        ChangeState(line_color=(0.5, 0.5, 0.5, 1.))
        Mesh(
            vertices=[
                -1 * sz, -1 * sz, -1 * sz,
                -1 * sz, -1 * sz, 1 * sz,
                -1 * sz, 1 * sz, 1 * sz,
                -1 * sz, 1 * sz, -1 * sz,
                1 * sz, -1 * sz, -1 * sz,
                1 * sz, -1 * sz, 1 * sz,
                1 * sz, 1 * sz, 1 * sz,
                1 * sz, 1 * sz, -1 * sz,
            ],
            indices=[0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(1., 0., 1., 1.))
        Mesh(
            vertices=[1 * sz, 0, 0, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(1., 1., 0., 1.))
        Mesh(
            vertices=[0, 1 * sz, 0, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(0., 1., 1., 1.))
        Mesh(
            vertices=[0, 0, 1 * sz, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(1.,1.,1.,1.))
        PopState()
        Color(1, 1, 1)
        # Rotate(-90, 1, 0, 0)
        # UpdateNormalMatrix()
        # Translate(0, 5, 0)
        self.container = InstructionGroup()
        PopMatrix()

    def prepare_test_unit(self, test=3):
        template = None
        if test == 1:
            template = 'unmoba2'
            if template not in self.models_data:
                self.models_data[template] = res.read_model_data('figures.res', template, 'models', save_json=False)
                unit = self.scene.create_unit_from_model_data(
                    self.models_data[template],
                    coefs=[0, 0, 0],
                    textures={'*': 'banshee02.png'},
                )
                return unit.name
        if test == 2:
            template = 'unmogo'
            if template not in self.models_data:
                self.models_data[template] = res.read_model_data('figures.res', template, 'models', save_json=False)
                unit = self.scene.create_unit_from_model_data(
                    self.models_data[template],
                    coefs=[0, 0, 0],
                    # selected_parts=['hp', 'bd', 'hd', ],
                    excluded_parts=['rh3.pike00', ],
                    selected_animations=[],
                    textures={'*': 'goblin01.png'},
                )
                return unit.name
        if test == 3:
            template = 'unhufe'
            if template not in self.models_data:
                self.models_data[template] = res.read_model_data('figures.res', template, 'models', save_json=False)
                unit = self.scene.create_unit_from_model_data(
                    self.models_data[template],
                    coefs=[0, 0, 0],
                    selected_parts=[
                        'hp',
                        'bd',
                        'hd',
                        'rh1',
                        'rh2',
                        'rh3',
                        'lh1',
                        'lh2',
                        'lh3',
                        'll1',
                        'll2',
                        'll3',
                        'rl1',
                        'rl2',
                        'rl3',
                        'hr.01',
                    ],
                    selected_animations=[
                        'cidle07',
                        'crun01',
                        'ccrawl01',
                        'crest',
                        'cspecial14',
                        'cwalk01',
                        'cwalk02',
                        'cwalk05',
                        'uattack01',
                        'uattack02',
                        'uattack08',
                        'ubriefing06',
                        'ucast03',
                        'ucross06',
                        'udeath06',
                        'uhit14',
                        'uspecial05',
                        'udeath15',
                    ],
                    textures={'*': 'unhufeskin_08.png'},
                )
                return unit.name
        for unit in self.scene.units.values():
            if unit.template == template:
                return unit.name
        return None


class EvilIslandsModelsApp(App):

    def build(self):
        return Renderer()


if __name__ == '__main__':
    if not os.path.isfile('figures.res'):
        print('Please copy "figures.res" file into the current folder')
        sys.exit(-1)
    EvilIslandsModelsApp().run()
