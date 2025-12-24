import os
import sys


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
        self.loaded = set()
        self.scene = dat.SceneData()
        self.models_data = {}
        self.prepare_scene()
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        self.canvas['texture_id'] = 1
        Clock.schedule_interval(self.update_glsl, 1 / 20.)
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
            u = list(self.scene.units.values())[0]
            current_animation_ind = u.animations_loaded.index(u.animation_playing)
            current_animation_ind += 1
            if current_animation_ind >= len(u.animations_loaded):
                current_animation_ind = 0
            u.animation_playing = u.animations_loaded[current_animation_ind]
            u.animation_frame = 0
            if _Debug:
                print(f'playing animation {u.animation_playing} for {u.name}')
        elif keycode[1] == 'x':
            u = list(self.scene.units.values())[0]
            current_animation_ind = u.animations_loaded.index(u.animation_playing)
            current_animation_ind -= 1
            if current_animation_ind < 0:
                current_animation_ind = len(u.animations_loaded) - 1
            u.animation_playing = u.animations_loaded[current_animation_ind]
            u.animation_frame = 0
            if _Debug:
                print(f'playing animation {u.animation_playing} for {u.name}')
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
                self.global_rotate_y.angle += ax
                self.global_rotate_x.angle += ay

    def add_mesh(self, name):
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
        self.loaded.add(name)
        mesh.onstage = True
        if _Debug:
            print(f'added mesh {name} on scene with {len(mesh.vertices)} vertices and {len(mesh.indices)} indices')

    def remove_mesh(self, name):
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.meshes:
            raise Exception(f'Mesh {name} does not exist')
        mesh = self.scene.meshes[name]
        self.container.remove_group(name)
        self.loaded.remove(name)
        mesh.onstage = False
        if _Debug:
            print(f'removed {name} from scene')

    def add_unit(self, name):
        if not self.container:
            raise Exception('Container was not ready')
        if name not in self.scene.units:
            raise Exception(f'Unit {name} does not exist')
        unit = self.scene.units[name]
        
        def _before(_, part_name):
            mesh = unit.meshes.get(part_name)
            mesh.part_bone = Transform(group=mesh.name)
            mesh.part_bone.translate(unit.parts_pos[part_name][0], unit.parts_pos[part_name][1], unit.parts_pos[part_name][2])
            mesh.part_rotate = Rotate(0, 1, 0, 0, group=mesh.name)
            self.container.add(PushMatrix(group=mesh.name))
            self.container.add(mesh.part_bone)
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

        def _after(_, part_name):
            mesh = unit.meshes.get(part_name)
            self.container.add(PopMatrix(group=mesh.name))
            self.loaded.add(name)
            mesh.onstage = True
            if _Debug:
                print(f'added mesh {name} on scene with {len(mesh.vertices)} vertices and {len(mesh.indices)} indices')

        unit.walk_parts(_before, _after)
        unit.onstage = True
        if _Debug:
            print(f'added unit {unit.name} on the stage')

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
        self.gl_error('step1')
        self.canvas['texture_id'] = 1
        self.canvas['projection_mat'] = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['modelview_mat'] = Matrix().look_at(
            0, 0, 2,
            0, 0, 0,
            0, 1, 0,
        )
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.gl_error('step2')
        # TODO: maintain separate list of active animations for all units
        for unit in self.scene.units.values():
            if not unit.animation_playing:
                continue
            parts_animations = unit.animations[unit.animation_playing]
            root_part_name = unit.parts[0]
            root_part_animation_info = parts_animations.get(root_part_name)
            if unit.animation_frame >= len(root_part_animation_info['rotation_frames']):
                if _Debug:
                    print(f'restart {unit.name} animation {unit.animation_playing} after frame {unit.animation_frame}')
                unit.animation_frame = 0
            # root_quaternion = root_part_animation_info['rotation_frames'][unit.animation_frame]
            for part_name in unit.parts:
                if part_name not in parts_animations:
                    continue
                part_animation_info = parts_animations.get(part_name)
                part_quaternion = part_animation_info['rotation_frames'][unit.animation_frame]
                mesh = unit.meshes[part_name]
                # calculated_quaternion = quaternion.quaternion_multiply(root_quaternion, part_quaternion)
                # calculated_quaternion = quaternion.quaternion_multiply(root_quaternion, calculated_quaternion)
                calculated_quaternion = part_quaternion
                # axis, angle = quaternion.quaternionToAxisAngle(calculated_quaternion)
                # mesh.part_rotate.set(math.degrees(angle), axis[0], axis[1], axis[2])
                rotate_mat = Matrix()
                rotate_mat.set(array=mth.quaternionToRotationMatrix(calculated_quaternion))
                mesh.part_rotate.matrix = rotate_mat.inverse()
            unit.animation_frame += 1

    def setup_scene(self):
        PushMatrix()
        self.global_translate = Translate(0, 0, 0)
        self.global_rotate_x = Rotate(0, 1, 0, 0)
        self.global_rotate_y = Rotate(0, 0, 1, 0)
        self.global_scale = Scale(1.0)
        sz = 1
        PushState()
        ChangeState(line_color=(1., 0., 0., 1.))
        Mesh(
            vertices=[1 * sz, 0, 0, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(0., 1., 0., 1.))
        Mesh(
            vertices=[0, 1 * sz, 0, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(0., 0., 1., 1.))
        Mesh(
            vertices=[0, 0, 1 * sz, 0, 0, 0],
            indices=[0, 1],
            fmt=[(b'v_pos', 3, 'float'), ],
            mode='lines',
        )
        ChangeState(line_color=(1., 1., 1., 0.9))
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
        ChangeState(line_color=(1.,1.,1.,1.))
        PopState()
        Color(1, 1, 1)
        # Rotate(-90, 1, 0, 0)
        # UpdateNormalMatrix()
        self.container = InstructionGroup()
        PopMatrix()

    def prepare_scene(self, test=3):
        if test == 1:
            self.models_data['unmoba2'] = res.read_model_data('figures.res', 'unmoba2', 'models', save_json=False)
            self.scene.create_unit_from_model_data(
                self.models_data['unmoba2'],
                coefs=[0, 0, 0],
                textures={'*': 'banshee02.png'},
            )
        elif test == 2:
            self.models_data['unhufe'] = res.read_model_data('figures.res', 'unhufe', 'models', save_json=False)
            self.scene.create_unit_from_model_data(
                self.models_data['unhufe'],
                coefs=[0, 0, 1],
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
                    'crun01',
                    'cidle07',
                    'ccrawl01',
                    'crest',
                    'cspecial14',
                    'cwalk01',
                    'cwalk02',
                    'cwalk03',
                    'cwalk04',
                    'cwalk05',
                    'cwalk06',
                    'cwalk07',
                    'uattack01',
                    'uattack02',
                    'uattack03',
                    'uattack04',
                    'uattack05',
                    'uattack06',
                    'uattack07',
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
        elif test == 3:
            self.models_data['unmogo'] = res.read_model_data('figures.res', 'unmogo', 'models', save_json=False)
            self.scene.create_unit_from_model_data(
                self.models_data['unmogo'],
                coefs=[1, 0, 0],
                excluded_parts=['rh3.pike00', ],
                selected_animations=[],
                textures={'*': 'goblin01.png'},
            )


class EvilIslandsModelsApp(App):

    def build(self):
        return Renderer()


if __name__ == '__main__':
    if not os.path.isfile('figures.res'):
        print('Please copy "figures.res" file into the current folder')
        sys.exit(-1)
    EvilIslandsModelsApp().run()
