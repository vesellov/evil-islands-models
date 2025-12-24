import res
import mth


_Debug = True


_NextUnitID = 0
_NextMeshID = 0


class MeshData(object):

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.unit_name = None
        self.unit_part_name = None
        self.onstage = False
        self.vertices = []
        self.indices = []
        self.material = kwargs.get('material', None)
        self.part_translate = None
        self.part_animate = None


class UnitData(object):

    def __init__(self):
        self.name = None
        self.template = None
        self.meshes = {}
        self.parts = []
        self.parts_tree = {}
        self.parts_pos = {}
        self.parts_parents = {}
        self.textures = {}
        self.animations = {}
        self.animations_loaded = []
        self.animation_playing = None
        self.animation_frame = 0
        self.onstage = False

    def list_parents(self, part_name):
        if part_name not in self.parts_parents:
            return []
        parents = []
        current_part = part_name
        while current_part and current_part in self.parts_parents:
            next_parent = self.parts_parents[current_part]
            if next_parent:
                parents.insert(0, next_parent)
            current_part = next_parent
        return parents

    def walk_parts(self, visitor_before, visitor_after, tree=None):
        if tree is None:
            tree = self.parts_tree
        for part_name, other_parts in tree.items():
            if part_name in self.parts:
                visitor_before(self, part_name)
                self.walk_parts(visitor_before, visitor_after, other_parts)
                visitor_after(self, part_name)


class SceneData(object):

    def __init__(self):
        self.units = {}
        self.meshes = {}
        # buffers
        self._file = None
        self._object = None
        self._vertices = []
        self._normals = []
        self._texcoords = []
        self._faces = []

    def create_mesh_from_fig_data(self, fig_data, prefix='', texture_filename=None, coefs=[0, 0, 0]):
        global _NextMeshID
        _NextMeshID += 1
        name = prefix + '_' + str(_NextMeshID)
        mesh = MeshData(
            name=name,
            material={'map_Kd': texture_filename} if texture_filename else None,
        )
        vert_buf = []
        norm_buf = []
        tex_buf = []
        for i in range(fig_data[1]):
            for j in range(4):
                vert_buf.append(mth.ei2xyz_list([
                    mth.trilinear([fig_data[13][i][0][k][j] for k in range(8)], coefs),
                    mth.trilinear([fig_data[13][i][1][k][j] for k in range(8)], coefs),
                    mth.trilinear([fig_data[13][i][2][k][j] for k in range(8)], coefs),
                ]))
        for i in range(fig_data[2]):
            for j in range(4):
                norm_buf.append(mth.ei2xyz_list([
                    fig_data[14][i][0][j],
                    fig_data[14][i][1][j],
                    fig_data[14][i][2][j],
                ]))
        for i in range(fig_data[3]):
            tex_buf.append(fig_data[15][i])
        idx = 0
        d = fig_data[17]
        for i in fig_data[16]:
            for f in range(3):
                j = i[f]
                mesh.vertices.extend([
                    vert_buf[d[j][0] * 4 + d[j][1]][0],
                    vert_buf[d[j][0] * 4 + d[j][1]][1],
                    vert_buf[d[j][0] * 4 + d[j][1]][2],
                    norm_buf[d[j][2] * 4 + d[j][3]][0],
                    norm_buf[d[j][2] * 4 + d[j][3]][1],
                    norm_buf[d[j][2] * 4 + d[j][3]][2],
                    tex_buf[d[j][4]][0],
                    tex_buf[d[j][4]][1],
                ])
            mesh.indices.extend([idx, idx + 1, idx + 2])
            idx += 3
        self.meshes[name] = mesh
        if _Debug:
            print(f'prepared mesh {name} with {idx} faces')
        return mesh
    
    def create_unit_from_model_data(self, data, coefs=[0, 0, 0], selected_parts=[], excluded_parts=[], selected_animations=[], textures={'*': 'default.png'}):
        global _NextUnitID
        _NextUnitID += 1
        u = UnitData()
        u.template = data['template']
        u.name = u.template + str(_NextUnitID)
        u.textures = textures
        parts_tree = data['links'][u.template+'.lnk']['info']
        u.parts_parents = data['links'][u.template+'.lnk']['parents']
        u.parts_tree = data['links'][u.template+'.lnk']['tree']
        ordered_parts_list = res.flat_tree(parts_tree)
        u.animations_loaded = selected_animations
        if not u.animations_loaded:
            u.animations_loaded = [a[:-4] for a in data['animations'].keys()]
        related_meshes = {}
        first_animation_name = None
        if _Debug:
            print(f'about to prepare {u.name} in {len(ordered_parts_list)} parts')
        if not selected_parts:
            selected_parts = ordered_parts_list
        for exclude in excluded_parts:
            if exclude in selected_parts:
                selected_parts.remove(exclude)
        for part_name in selected_parts:
            u.parts.append(part_name)
            part_info = data['bones'][part_name + '.bon']
            u.parts_pos[part_name] = mth.ei2xyz_list([
                mth.trilinear([part_info[i][0] for i in range(8)], coefs),
                mth.trilinear([part_info[i][1] for i in range(8)], coefs),
                mth.trilinear([part_info[i][2] for i in range(8)], coefs),
            ])
            mesh = self.create_mesh_from_fig_data(
                fig_data=data['figures'][part_name+'.fig'],
                prefix=u.name + '_' + part_name,
                texture_filename=u.textures[part_name] if part_name in u.textures else u.textures['*'],
                coefs=coefs,
            )
            mesh.unit_name = u.name
            mesh.unit_part_name = part_name
            u.meshes[part_name] = mesh
            related_meshes[part_name] = mesh.name
            part_animations_loaded = []
            for anim_name in u.animations_loaded:
                if part_name not in data['animations'][anim_name+'.anm']:
                    continue
                animation_info = data['animations'][anim_name+'.anm'][part_name]
                if anim_name not in u.animations:
                    u.animations[anim_name] = {}
                u.animations[anim_name][part_name] = {
                    'rotation_frames': [mth.ei2quad_list(quad) for quad in animation_info[1]],
                    'translation_frames': [mth.ei2xyz_list(coord) for coord in animation_info[3]],
                }
                if animation_info[4] != 0 and animation_info[5] != 0:
                    morphing_frames = []
                    for value in animation_info[6]:
                        morphing_frames.append([])
                        for i in range(animation_info[5]):
                            morphing_frames[0].append(mth.ei2xyz_list(value[i]))
                    u.animations[anim_name][part_name]['morphing_frames'] = morphing_frames
                if not first_animation_name:
                    first_animation_name = anim_name
                part_animations_loaded.append(anim_name)
            if part_animations_loaded:
                if _Debug:
                    print(f'prepared {len(part_animations_loaded)} animations for {part_name}')
        u.animation_playing = first_animation_name
        self.units[u.name] = u
        return u
