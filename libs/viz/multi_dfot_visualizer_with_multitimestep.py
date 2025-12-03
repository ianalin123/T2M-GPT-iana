import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import argparse
import dataclasses
from typing import List, Literal, Tuple

import libs.utils.fncsmpl as fncsmpl
import torch
import viser
from libs.utils.transforms import SE3, SO3
from viser import transforms as tf

from color_gradation import SLAHMR_COLORS, gradient_oklab, oklab_to_rgb, rgb_to_oklab, make_oklab_shade_fn


class DataProcessor:
    @staticmethod
    def get_color(person_num, sample_num):
        gt_color_list = [
            # (192, 57, 43), # GT 1 - deep red
            # (41, 128, 185), # GT 2 - strong blue
            (150, 52, 61),
            (34, 74, 107),
            (63, 106, 88),
            (127, 78, 119),
            # (210, 126, 141),
            # (107, 148, 194),
            # (196, 110, 125),
            # (95, 134, 182),
            (236, 136, 236),  # GT 3 - pink
            (133, 210, 233),  # GT 4 - light blue
            (255, 85, 0),  # GT 5 - orange
        ]
        gt_colors = [[gt_color_list[i] for i in range(person_num)]]


        def rose_gradation(i):
            START = (226, 87, 73)
            END = (231, 156, 141)
            # return gradient_oklab(START, END, i, sample_num)
            fn = make_oklab_shade_fn(START, limit=0.8, curve=0.8)
            return fn(0.1*i)

        def sky_gradation(i):
            START = (44, 108, 161)
            END = (140, 186, 195)  # (189,226,210)
            # return gradient_oklab(START, END, i, sample_num)
            fn = make_oklab_shade_fn(START, limit=0.8, curve=0.8)
            return fn(0.1*i)

        def mint_gradation(i):
            START = (46, 92, 73)
            END = (127, 180, 161)
            # return gradient_oklab(START, END, i, sample_num)
            fn = make_oklab_shade_fn(START, limit=0.8, curve=0.8)
            return fn(0.1*i)

        def plum_gradation(i):
            START = (106, 58, 102)
            END = (169, 127, 165)
            # return gradient_oklab(START, END, i, sample_num)
            fn = make_oklab_shade_fn(START, limit=0.8, curve=0.8)
            return fn(0.1*i)

        # gradation_func = [yellow_gradation, green_gradation, purple_gradation, pink_gradation, emerald_gradation]
        gradation_func = [rose_gradation, sky_gradation, mint_gradation, plum_gradation]

        pred_colors = [
            [gradation_func[j](i) for j in range(person_num)] for i in range(sample_num)
        ]
        print(pred_colors)
        colors = gt_colors + pred_colors
        return colors

    @staticmethod
    def get_person_name(person_num, sample_idx_list):
        person_names = ["Alice", "Bob", "Charlie", "David", "Eve"]
        gt_person_names = [[f"GT/{person_names[i]}" for i in range(person_num)]]
        pred_person_names = [
            [f"Pred/Sample_{i}/{person_names[j]}" for j in range(person_num)]
            for i in sample_idx_list
        ]
        person_names = gt_person_names + pred_person_names
        return person_names


class Color:
    start_idx: int
    rgb: Tuple[int, int, int]


class VizData:
    smpl_posed_all: fncsmpl.SmplShapedAndPosed | None = None
    smpl_shaped_all: fncsmpl.SmplShaped | None = None
    T_world_root_all: np.ndarray | None = None
    Ts_world_joint_all: np.ndarray | None = None

    curr_body_handles: List[List[viser.MeshSkinnedHandle]] = []
    prev_body_handles: List[List[viser.MeshSkinnedHandle]] = []
    multi_timestep_handles: List[
        List[List[viser.MeshSkinnedHandle]]
    ] = []  # [timestep_idx][sample_idx][person_idx]

    # body_model: fncsmpl.SmplModel = fncsmpl.SmplModel.load('./data/smplh/neutral/model.npz')
    body_model: fncsmpl.SmplModel = fncsmpl.SmplModel.load(
        "./data/smplx/SMPLX_NEUTRAL.npz"
    ).to("cuda")
    skin_weights: np.ndarray | None = None

    timesteps: int = 10
    context_timesteps: int = 0
    sample_idx_list: List[int] = [0, 1]
    all_sample_num: int = 0
    colors: List[List[Tuple[int, int, int]]] = []
    hand_pose_type: Literal["Zero", "Fist"] = "Fist"
    mode: Literal["Playback", "Multitimestep"] = "Playback"
    visible_timesteps: int = 16
    selected_timesteps: List[int] = []

    gt_timesteps: int = 0

    def collapse_weights_to_top3_parent_pool(self, W, parents, eps=1e-8):
        parents = [0] + [i + 1 for i in parents]
        V, B = W.shape
        W_new = W.copy()
        for v in range(V):
            active = np.nonzero(W_new[v] > eps)[0].tolist()
            while len(active) > 3:
                j = min(active, key=lambda i: W_new[v, i])
                p = parents[j]
                if p == -1:
                    p = np.argmax(W_new[v])
                W_new[v, p] += W_new[v, j]
                W_new[v, j] = 0.0
                active.remove(j)
            s = W_new[v].sum()
            if s > 0:
                W_new[v] /= s
        return W_new

    def remove_prev_data(self) -> None:
        for prev_body_handle_list in self.prev_body_handles:
            for prev_body_handle in prev_body_handle_list:
                prev_body_handle.remove()
        self.prev_body_handles = []

    def invisible_data(self) -> None:
        for body_handles in self.curr_body_handles:
            for body_handle in body_handles:
                body_handle.visible = False
        for body_handles in self.prev_body_handles:
            for body_handle in body_handles:
                body_handle.visible = False

    def change_visible_data(
        self,
    ) -> None:
        for idx, body_handles in enumerate(self.curr_body_handles):
            for body_handle in body_handles:
                if idx in self.sample_idx_list:
                    body_handle.visible = True
                else:
                    body_handle.visible = False
        for body_handles in self.prev_body_handles:
            for body_handle in body_handles:
                body_handle.visible = False

    def create_hand_poses(
        self,
        B: int,
        T: int,
        P: int,
        device: torch.device,
        pose_type: Literal["Zero", "Fist"] = "Fist",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Create left and right hand poses from predefined axis-angle rotations.

        Right hand is mirrored by negating the Z component of the axis-angle.

        Args:
            B: Batch size
            T: Timesteps
            P: Number of persons
            device: Device to create tensors on
            pose_type: Type of hand pose - "Zero" or "Fist"

        Returns:
            Tuple of (left_hand_quats, right_hand_quats), or (None, None) if pose_type is "Zero"
        """
        if pose_type == "Zero":
            return None, None

        # Joint angles in axis-angle format (SMPLX joints 25-39 -> hand indices 0-14)
        left_hand_axis_angles = torch.tensor(
            [
                [0.11, 0.17, -1.16],  # joint 25 -> hand idx 0
                [-0.00, -0.00, -0.97],  # joint 26 -> hand idx 1
                [0.00, 0.00, -1.12],  # joint 27 -> hand idx 2
                [-0.26, 0.05, -1.54],  # joint 28 -> hand idx 3
                [0.00, 0.00, 0.00],  # joint 29 -> hand idx 4
                [-0.13, -0.10, -1.77],  # joint 30 -> hand idx 5
                [-0.42, -0.56, -1.16],  # joint 31 -> hand idx 6
                [-0.20, -0.45, -0.58],  # joint 32 -> hand idx 7
                [-1.35, 0.04, -0.39],  # joint 33 -> hand idx 8
                [-0.20, -0.28, -1.34],  # joint 34 -> hand idx 9
                [-0.18, -0.00, -0.54],  # joint 35 -> hand idx 10
                [-0.50, -0.13, -1.25],  # joint 36 -> hand idx 11
                [1.23, 0.32, 0.33],  # joint 37 -> hand idx 12
                [0.08, -0.72, -0.49],  # joint 38 -> hand idx 13
                [-0.44, 1.29, -1.15],  # joint 39 -> hand idx 14
            ],
            device=device,
            dtype=torch.float32,
        )  # Shape: (15, 3)

        # Mirror for right hand by negating Y and Z components
        right_hand_axis_angles = left_hand_axis_angles.clone()
        right_hand_axis_angles[:, 1] *= -1  # Negate Y
        right_hand_axis_angles[:, 2] *= -1  # Negate Z

        # Convert all axis-angles to quaternions in one batch operation
        left_hand_quats_flat = SO3.exp(left_hand_axis_angles).wxyz  # Shape: (15, 4)
        right_hand_quats_flat = SO3.exp(right_hand_axis_angles).wxyz  # Shape: (15, 4)

        # Expand to (B, T, P, 15, 4)
        left_hand_quats = (
            left_hand_quats_flat.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T, P, 15, 4)
        )
        right_hand_quats = (
            right_hand_quats_flat.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T, P, 15, 4)
        )

        return left_hand_quats, right_hand_quats

    def update_smpl_data(
        self,
        betas: torch.Tensor,
        T_world_root: torch.Tensor,
        body_joint_rotations: torch.Tensor,
        context_timesteps: int = 0,
        gt_timesteps: int = 0,
    ) -> None:
        print("update_smpl_data start")
        self.smpl_shaped_all = self.body_model.with_shape(betas)

        # Create hand poses (right hand is mirrored version of left)
        B, T, P = T_world_root.shape[:3]
        left_hand_quats, right_hand_quats = self.create_hand_poses(
            B, T, P, T_world_root.device, self.hand_pose_type
        )

        self.smpl_posed_all = self.smpl_shaped_all.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_joint_rotations,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
        )

        self.T_world_root_all = self.smpl_posed_all.T_world_root.numpy(force=True)
        self.Ts_world_joint_all = self.smpl_posed_all.Ts_world_joint.numpy(force=True)
        self.Ts_world_joint_all[..., 4:] -= self.T_world_root_all[..., None, 4:]

        self.timesteps = T_world_root.shape[1]
        self.context_timesteps = context_timesteps
        self.gt_timesteps = gt_timesteps
        self.all_sample_num = betas.shape[0] - 1
        self.person_num = betas.shape[2]

        self.person_names = DataProcessor.get_person_name(
            self.person_num, range(self.all_sample_num)
        )
        self.colors = DataProcessor.get_color(
            self.person_num, len(self.sample_idx_list[1:])
        )
        print("update_smpl_data end")

    def update_sample_idx_list(
        self,
        sample_idx_list: List[int],
        scene: viser.SceneApi,
    ) -> None:
        print(f"update sample idx list: {sample_idx_list[1:]}")
        self.sample_idx_list = sample_idx_list
        self.colors = DataProcessor.get_color(self.person_num, len(sample_idx_list[1:]))
        self.change_visible_data()

    def add_body_handles_to_scene(
        self,
        scene: viser.SceneApi,
        show_gt: bool = False,
    ) -> None:
        self.invisible_data()
        self.prev_body_handles.extend(self.curr_body_handles)

        self.curr_body_handles = [
            self.create_body_handles(sample_idx, person_name_list, scene, show_gt)
            for sample_idx, person_name_list in enumerate(
                self.person_names,
            )
        ]

    def create_body_handles(
        self,
        sample_idx: int,
        person_name_list: List[str],
        scene: viser.SceneApi,
        show_gt: bool = False,
    ) -> List[viser.MeshSkinnedHandle]:
        if self.skin_weights is None:
            self.skin_weights = self.collapse_weights_to_top3_parent_pool(
                self.body_model.weights.numpy(force=True),
                list(self.body_model.parent_indices),
            )

        # GT (sample_idx 0) should only be visible if show_gt is True
        if sample_idx == 0:
            is_visible = show_gt and sample_idx in self.sample_idx_list
        else:
            is_visible = sample_idx in self.sample_idx_list

        handles = [
            scene.add_mesh_skinned(
                f"/persons/{person_name_list[person_idx]}/{uuid.uuid4()}",
                vertices=self.smpl_shaped_all.verts_zero[
                    sample_idx, 0, person_idx, :
                ].numpy(force=True),
                faces=self.body_model.faces.numpy(force=True),
                bone_wxyzs=tf.SO3.identity(
                    batch_axes=(self.body_model.get_num_joints() + 1,)
                ).wxyz,
                bone_positions=np.concatenate(
                    [
                        np.zeros((1, 3)),
                        self.smpl_shaped_all.joints_zero[
                            sample_idx, 0, person_idx, :
                        ].numpy(force=True),
                    ],
                    axis=0,
                ),
                skin_weights=self.skin_weights,
                visible=is_visible,
            )
            for person_idx in range(self.person_num)
        ]
        return handles


def load_visualizer(server, data_root_dir):
    raw_data = None
    is_update_ok = False
    viz_data = VizData()

    def get_subdir_list():
        subdirs = []
        for i in data_root_dir.glob("*"):
            if i.is_dir():
                subdirs.append(str(i.relative_to(data_root_dir)))
        return ["None"] + sorted(subdirs)

    def get_file_list(dir=data_root_dir):
        return ["None"] + sorted(str(p.relative_to(dir)) for p in dir.glob("**/*.npz"))

    # add ground and lights BEFORE creating tabs (so lights can be referenced in Lights tab)
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/ground",
        plane="xz",
        width=100,
        height=100,
        # cell_color=(80, 80, 80),
        # section_color=(250, 250, 250),
        cell_size=0.5,
        cell_thickness=0.5,
        section_size=1.0,
        section_thickness=0.7,
        position=(0.0, 0.0, 0.0),
        infinite_grid=False,
        fade_distance=30.0,
    )
    server.scene.add_light_hemisphere(
        name="light_hemisphere",
        sky_color=(255, 255, 255),
        ground_color=(200, 200, 200),
        intensity=1.0,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        visible=True,
    )

    # add timestep handles
    timestep_handles = [server.scene.add_frame("timesteps/0", show_axes=False)]

    # Create tabs for organization
    file_tab = server.gui.add_tab_group()

    # FILE TAB
    with file_tab.add_tab("File"):
        subdir_dropdown = server.gui.add_dropdown("Dir", options=get_subdir_list())
        file_dropdown = server.gui.add_dropdown("File", options=get_file_list())
        refresh_file_list = server.gui.add_button("Refresh File List")
        sample_num_dropdown = server.gui.add_dropdown(
            "Sample Num", initial_value=str(1), options=[str(i) for i in range(0, 11)]
        )

        sample_idx_folder = server.gui.add_folder("Sample Idx", visible=True)
        sample_check_buttons = []
        with sample_idx_folder:
            for sample_idx in range(10):
                if sample_idx == 0:
                    sample_check_buttons.append(
                        server.gui.add_checkbox(
                            f"idx_{sample_idx + 1:02d}", True, visible=False
                        )
                    )
                else:
                    sample_check_buttons.append(
                        server.gui.add_checkbox(
                            f"idx_{sample_idx + 1:02d}", False, visible=False
                        )
                    )

    # DISPLAY TAB
    with file_tab.add_tab("Display"):
        mode_dropdown = server.gui.add_dropdown(
            "Mode",
            initial_value="Playback",
            options=["Playback", "Multitimestep"],
        )

        sample_color_mode_dropdown = server.gui.add_dropdown(
            "Sample color mode",
            initial_value="Uniform",
            options=(
                "Uniform",
                "Trajectory (idx)",
                "Trajectory (rgb)",
                "Inbetween",
            ),
        )

        # Manual color selection folder
        idx_color_folder = server.gui.add_folder("Trajectory Colors", visible=False)
        idx_color_sliders = {}  # Will store sliders as {(sample_idx, person_idx): slider}

        # Trajectory color selection folder
        trajectory_colors_folder = server.gui.add_folder(
            "Trajectory Colors", visible=False
        )
        trajectory_rgb_pickers = {}  # {(sample_idx, person_idx): rgb_picker}
        inbetween_colors_folder = server.gui.add_folder("Inbetween Colors", visible=False)
        inbetween_rgb_pickers = {}  # {(sample_idx, person_idx): rgb_picker}
        # GT color selection folder
        gt_colors_folder = server.gui.add_folder("GT Colors", visible=True)
        gt_rgb_pickers = {}  # {person_idx: rgb_picker}

        @sample_color_mode_dropdown.on_update
        def _(_):
            idx_color_folder.visible = (
                sample_color_mode_dropdown.value == "Trajectory (idx)"
            )
            trajectory_colors_folder.visible = (
                sample_color_mode_dropdown.value == "Trajectory (rgb)"
            )
            inbetween_colors_folder.visible = (
                sample_color_mode_dropdown.value == "Inbetween"
            )

        show_gt_checkbox = server.gui.add_checkbox("Show GT", False)
        show_samples_checkbox = server.gui.add_checkbox("Show samples", True)
        hand_pose_dropdown = server.gui.add_dropdown(
            "Hand pose", initial_value="Fist", options=["Zero", "Fist"]
        )

        # Create meshes for all timesteps
        playback_folder = server.gui.add_folder("Playback", visible=True)
        with playback_folder:
            gui_timestep = server.gui.add_slider(
                "Timestep",
                min=0,
                max=viz_data.timesteps - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_start_end = server.gui.add_multi_slider(
                "Start/end",
                min=0,
                max=viz_data.timesteps - 1,
                initial_value=(0, viz_data.timesteps - 1),
                step=1,
            )
            gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
            gui_playing = server.gui.add_checkbox("Playing", False)
            gui_framerate = server.gui.add_slider(
                "FPS", min=1, max=120, step=1, initial_value=30
            )
            gui_framerate_options = server.gui.add_button_group(
                "FPS options", ("15", "30", "60", "120")
            )
            gui_gt_shift_slider = server.gui.add_slider(
                "GT distance",
                min=0,
                max=5,
                step=0.5,
                initial_value=2.0,
            )
            gui_shift_slider = server.gui.add_slider(
                "Sample distance",
                min=0,
                max=5,
                step=0.5,
                initial_value=0.0,
            )
            gui_motion_trails_playback = server.gui.add_checkbox("Motion trails", False)

        # Multitimestep folder
        multitimestep_folder = server.gui.add_folder("Multitimestep", visible=False)
        multitimestep_sliders = []
        multitimestep_checkboxes = []
        timestep_sliders_folder = None  # Will be created dynamically
        with multitimestep_folder:
            gui_num_visible_timesteps = server.gui.add_number(
                "# visible timesteps",
                initial_value=16,
                min=1,
                max=100,
                step=1,
            )
            gui_gt_shift_vector = server.gui.add_vector3(
                "GT offset",
                initial_value=(2.0, 0.0, 0.0),
                step=0.1,
            )
            gui_shift_vector = server.gui.add_vector3(
                "Sample offset",
                initial_value=(0.0, 0.0, 0.0),
                step=0.1,
            )
            gui_motion_trails_multi = server.gui.add_checkbox("Motion trails", False)

            # Timestep offset control
            gui_timestep_offset_vector = server.gui.add_vector3(
                "Offset per timestep",
                initial_value=(0.0, 0.0, 0.0),
                step=0.001,
            )

            # Lightness controls for gradient
            lightness_folder = server.gui.add_folder("Lightness / fade with timestep")
            with lightness_folder:
                gui_trail_start_brightness = server.gui.add_number(
                    "Trail start (oldest)",
                    initial_value=1.3,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )
                gui_trail_end_brightness = server.gui.add_number(
                    "Trail end (newest)",
                    initial_value=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )
                gui_mesh_start_brightness = server.gui.add_number(
                    "Mesh start (oldest)",
                    initial_value=1.3,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )
                gui_mesh_end_brightness = server.gui.add_number(
                    "Mesh end (newest)",
                    initial_value=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                )

            # timestep_sliders_folder will be created dynamically in create_timestep_sliders()

    # add ground
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/ground",
        plane="xz",
        width=100,
        height=100,
        # cell_color=(80, 80, 80),
        # section_color=(250, 250, 250),
        cell_size=0.5,
        cell_thickness=0.5,
        section_size=1.0,
        section_thickness=0.7,
        position=(0.0, 0.0, 0.0),
        infinite_grid=False,
        fade_distance=30.0,
    )
    server.scene.add_light_hemisphere(
        name="light_hemisphere",
        sky_color=(255, 255, 255),
        ground_color=(200, 200, 200),
        intensity=1.0,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        visible=True,
    )

    # add timestep handles
    timestep_handles = [server.scene.add_frame("timesteps/0", show_axes=False)]

    def create_multitimestep_handles(num_timesteps):
        """Create body handles for multitimestep mode.

        Args:
            num_timesteps: Number of timestep handles to create (must match number of sliders)
        """
        # Clear old handles
        for timestep_handles_list in viz_data.multi_timestep_handles:
            for sample_handles in timestep_handles_list:
                for handle in sample_handles:
                    try:
                        handle.remove()
                    except:
                        pass  # Handle already removed
        viz_data.multi_timestep_handles.clear()

        # Create handles for each visible timestep
        for timestep_idx in range(num_timesteps):
            timestep_handles = []
            for sample_idx in range(len(viz_data.curr_body_handles)):
                sample_handles = []
                for person_idx in range(viz_data.person_num):
                    # Create a body handle for this timestep/sample/person combination
                    person_name = viz_data.person_names[sample_idx][person_idx]
                    handle = server.scene.add_mesh_skinned(
                        f"/multitimestep/{timestep_idx}/{person_name}/{uuid.uuid4()}",
                        vertices=viz_data.smpl_shaped_all.verts_zero[
                            sample_idx, 0, person_idx, :
                        ].numpy(force=True),
                        faces=viz_data.body_model.faces.numpy(force=True),
                        bone_wxyzs=tf.SO3.identity(
                            batch_axes=(viz_data.body_model.get_num_joints() + 1,)
                        ).wxyz,
                        bone_positions=np.concatenate(
                            [
                                np.zeros((1, 3)),
                                viz_data.smpl_shaped_all.joints_zero[
                                    sample_idx, 0, person_idx, :
                                ].numpy(force=True),
                            ],
                            axis=0,
                        ),
                        skin_weights=viz_data.skin_weights,
                        visible=False,
                    )
                    sample_handles.append(handle)
                timestep_handles.append(sample_handles)
            viz_data.multi_timestep_handles.append(timestep_handles)

    def create_timestep_sliders():
        """Create or recreate timestep sliders based on # visible timesteps."""
        nonlocal multitimestep_sliders
        nonlocal multitimestep_checkboxes
        nonlocal timestep_sliders_folder

        # Remove old folder entirely (this removes all children automatically)
        if timestep_sliders_folder is not None:
            try:
                timestep_sliders_folder.remove()
            except:
                pass

        multitimestep_sliders.clear()
        multitimestep_checkboxes.clear()

        # Create new sliders and checkboxes
        num_sliders = int(gui_num_visible_timesteps.value)
        viz_data.visible_timesteps = num_sliders

        # Recreate the folder inside multitimestep_folder
        with multitimestep_folder:
            timestep_sliders_folder = server.gui.add_folder("Timestep Sliders")

        with timestep_sliders_folder:
            for i in range(num_sliders):
                # Calculate uniformly spaced initial value
                if num_sliders == 1:
                    initial_value = 0
                else:
                    initial_value = int(
                        i * (viz_data.timesteps - 1) / (num_sliders - 1)
                    )

                # Create checkbox
                checkbox = server.gui.add_checkbox(f"Show {i + 1}", True)
                multitimestep_checkboxes.append(checkbox)

                # Create slider
                slider = server.gui.add_slider(
                    f"Timestep {i + 1}",
                    min=0,
                    max=viz_data.timesteps - 1,
                    step=1,
                    initial_value=initial_value,
                )
                multitimestep_sliders.append(slider)

        # Update selected timesteps
        viz_data.selected_timesteps = [slider.value for slider in multitimestep_sliders]

        # Recreate multitimestep handles with the exact number we just created sliders for
        if viz_data.skin_weights is not None:
            create_multitimestep_handles(num_sliders)

    def update_color_controls():
        """Update color controls based on current sample_idx_list."""
        person_names = ["Alice", "Bob", "Charlie", "David", "Eve"]

        # Update manual color sliders
        for slider in idx_color_sliders.values():
            slider.remove()
        idx_color_sliders.clear()

        with idx_color_folder:
            for sample_idx in viz_data.sample_idx_list:
                if sample_idx == 0:  # Skip GT
                    continue
                for person_idx in range(viz_data.person_num):
                    slider = server.gui.add_slider(
                        f"Sample {sample_idx} - {person_names[person_idx]}",
                        min=0,
                        max=len(SLAHMR_COLORS) - 1,
                        step=1,
                        initial_value=(sample_idx * 7 + person_idx * 13)
                        % len(SLAHMR_COLORS),
                    )
                    idx_color_sliders[(sample_idx, person_idx)] = slider

        # Update trajectory color controls (one per sample per person)
        for picker in trajectory_rgb_pickers.values():
            picker.remove()
        trajectory_rgb_pickers.clear()

        with trajectory_colors_folder:
            for sample_idx in viz_data.sample_idx_list:
                if sample_idx == 0:  # Skip GT
                    continue
                for person_idx in range(viz_data.person_num):
                    # Get sample_num for color indexing
                    visible_samples = [s for s in viz_data.sample_idx_list if s > 0]
                    if sample_idx in visible_samples:
                        sample_num = visible_samples.index(sample_idx) + 1
                    else:
                        sample_num = 1

                    # Use colors from viz_data.colors
                    sample_num_ = min(sample_num, len(viz_data.colors) - 1)
                    initial_trajectory_color = viz_data.colors[sample_num_][person_idx]

                    rgb_picker = server.gui.add_rgb(
                        f"Sample {sample_idx} - {person_names[person_idx]}",
                        initial_value=initial_trajectory_color,
                    )

                    trajectory_rgb_pickers[(sample_idx, person_idx)] = rgb_picker

                    rgb_picker = server.gui.add_rgb(
                        f"Sample {sample_idx} - {person_names[person_idx]}",
                        initial_value=initial_trajectory_color,
                    )
                    inbetween_rgb_pickers[(sample_idx, person_idx)] = rgb_picker

    def load_data():
        nonlocal raw_data
        if raw_data is None:
            return

        nonlocal viz_data

        max_sample = 10
        if "body_joint_rotations" in raw_data:
            body_joint_rotations = (
                torch.from_numpy(raw_data["body_joint_rotations"][: max_sample + 1])
                .float()
                .to("cuda")
            )
        else:
            body_joint_rotations = (
                torch.from_numpy(raw_data["joint_rotations"][: max_sample + 1])
                .float()
                .to("cuda")
            )

        T_world_root = (
            torch.from_numpy(raw_data["T_world_root"][: max_sample + 1])
            .float()
            .to("cuda")
        )
        betas = torch.from_numpy(raw_data["betas"][: max_sample + 1]).float().to("cuda")
        # context_timesteps = raw_data["context_timesteps"]
        if "gt_timesteps" in raw_data:
            gt_timesteps = raw_data["gt_timesteps"]
        else:
            gt_timesteps = 0

        viz_data.update_smpl_data(
            betas,
            T_world_root,
            body_joint_rotations,
            0,
            gt_timesteps,
        )
        viz_data.add_body_handles_to_scene(server.scene, show_gt=show_gt_checkbox.value)

        for sample_idx, sample_check_button in enumerate(sample_check_buttons):
            if sample_idx < viz_data.all_sample_num:
                sample_check_button.visible = True
            else:
                break

        person_names = ["Alice", "Bob", "Charlie", "David", "Eve"]

        # Create GT color controls (these are created once and don't change)
        for picker in gt_rgb_pickers.values():
            picker.remove()
        gt_rgb_pickers.clear()

        with gt_colors_folder:
            for person_idx in range(viz_data.person_num):
                # Use GT colors from viz_data.colors[0] as initial values
                # initial_gt_color = viz_data.colors[0][person_idx]

                # Use a pink color with lightness based on person_idx
                L, a, b = rgb_to_oklab(255, 0, 255)
                L += person_idx / 8.0

                initial_gt_color = oklab_to_rgb(L, a, b)

                rgb_picker = server.gui.add_rgb(
                    f"{person_names[person_idx]} GT", initial_value=initial_gt_color
                )

                def make_gt_callback(person_idx, picker):
                    @picker.on_update
                    def _(_):
                        viz_data.colors[0][person_idx] = picker.value

                    return _

                make_gt_callback(person_idx, rgb_picker)
                gt_rgb_pickers[person_idx] = rgb_picker

        # Update color controls for visible samples (this will also create trajectory controls)
        update_color_controls()

        gui_timestep.max = viz_data.timesteps - 1
        gui_start_end.max = viz_data.timesteps - 1
        gui_start_end.value = (0, viz_data.timesteps - 1)

        # Initialize skin weights if not already done
        if viz_data.skin_weights is None and viz_data.curr_body_handles:
            viz_data.skin_weights = viz_data.collapse_weights_to_top3_parent_pool(
                viz_data.body_model.weights.numpy(force=True),
                list(viz_data.body_model.parent_indices),
            )

        # Initialize multitimestep sliders and handles
        create_timestep_sliders()

    def get_color_gradient(
        base_color: Tuple[int, int, int],
        num_steps: int,
        start_lightness: float = 1.0,
        end_lightness: float = 0.2,
    ) -> List[Tuple[int, int, int]]:
        """Generate a color gradient adjusting lightness in OKLAB space for perceptually uniform transitions.

        Args:
            base_color: Base RGB color tuple
            num_steps: Number of gradient steps
            start_lightness: Lightness factor for oldest position (0.0 = black, 1.0 = original, 2.0 = white)
            end_lightness: Lightness factor for newest position (0.0 = black, 1.0 = original, 2.0 = white)
        """
        # Convert base color to OKLAB
        L_base, a_base, b_base = rgb_to_oklab(*base_color)

        # Convert lightness factors to actual L values in OKLAB space
        # Factor 0.0 -> L=0 (black), Factor 1.0 -> L=L_base (original), Factor 2.0 -> L=1.0 (white)
        def factor_to_L(factor: float, L_base: float) -> float:
            if factor <= 1.0:
                # 0.0 to 1.0: interpolate from black (L=0) to original (L=L_base)
                return L_base * factor
            else:
                # 1.0 to 2.0: interpolate from original (L=L_base) to white (L=1.0)
                alpha = factor - 1.0  # 0.0 to 1.0
                return L_base + (1.0 - L_base) * alpha

        L_start = factor_to_L(start_lightness, L_base)
        L_end = factor_to_L(end_lightness, L_base)

        colors = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0

            # Linearly interpolate L in OKLAB space (perceptually uniform)
            L = L_start + (L_end - L_start) * t

            # For chroma (a, b), we need to scale based on how far we are from extremes
            # When at black (L=0) or white (L=1.0), chroma should be 0
            # At the base lightness, chroma should be at base values
            if L_base > 0:
                # Calculate how much chroma to preserve based on distance from black/white
                if L <= L_base:
                    # Between black and base: scale chroma proportionally
                    chroma_scale = L / L_base if L_base > 0 else 0
                else:
                    # Between base and white: scale chroma down as we approach white
                    chroma_scale = (1.0 - L) / (1.0 - L_base) if L_base < 1.0 else 0

                a = a_base * chroma_scale
                b = b_base * chroma_scale
            else:
                a = 0
                b = 0

            colors.append(oklab_to_rgb(L, a, b))

        return colors

    # Track trail handles and previous state to avoid unnecessary updates
    trail_handles = {}
    prev_multitimestep_state = None  # Will store (selected_timesteps, checkbox_states, gt_shift, sample_shift, motion_trails)

    def update_body_at_timestep(sample_idx, sample_num, person_idx, t, offset):
        """Update a body handle to show pose at timestep t."""
        zero_pos = np.zeros(3, dtype=np.float32)
        sample_body_handle = viz_data.curr_body_handles[sample_idx]
        body_handle = sample_body_handle[person_idx]

        # Make sure body is visible (respecting sample_idx_list)
        if sample_idx in viz_data.sample_idx_list:
            body_handle.visible = True

        if not body_handle.visible:
            return

        if viz_data.timesteps <= t:
            return

        T_world_root = viz_data.T_world_root_all[sample_idx, t, person_idx]
        body_handle.bones[0].position = zero_pos
        body_handle.bones[0].wxyz = T_world_root[:4]
        Ts_world_joint = viz_data.Ts_world_joint_all[sample_idx, t, person_idx]
        for b, bone_handle in enumerate(body_handle.bones[1:]):
            bone_transform = Ts_world_joint[b]
            bone_handle.position = bone_transform[4:7]
            bone_handle.wxyz = bone_transform[:4]
        body_handle.position = offset + T_world_root[4:7]

        # GT always uses GT color from RGB picker
        if sample_idx == 0:
            # if t <= viz_data.gt_timesteps:
            if t <= viz_data.context_timesteps:
                if person_idx in gt_rgb_pickers:
                    body_handle.color = gt_rgb_pickers[person_idx].value
                else:
                    body_handle.color = viz_data.colors[0][person_idx]
            else:
                if sample_color_mode_dropdown.value == "Inbetween":
                    if t <4 or t > viz_data.timesteps - 5:
                        body_handle.color = gt_rgb_pickers[person_idx].value
                    else:
                        body_handle.color = (128, 128, 128)
                else:
                    body_handle.color = (128, 128, 128)
        elif t < viz_data.context_timesteps:
            # Samples in context period: use GT color from RGB picker
            
            if sample_color_mode_dropdown.value == "Inbetween":
                    if t <4 or t > viz_data.timesteps - 5:
                       body_handle.color = gt_rgb_pickers[person_idx].value
                    else:
                        body_handle.color = viz_data.colors[0][person_idx]
            else:
              if person_idx in gt_rgb_pickers:
                  body_handle.color = gt_rgb_pickers[person_idx].value
              else:
                  body_handle.color = viz_data.colors[0][person_idx]
        else:
            # Determine color based on sample color mode
            color_mode = sample_color_mode_dropdown.value
            if color_mode == "Uniform":
                # All samples use first sample color
                body_handle.color = viz_data.colors[1][person_idx]
            elif color_mode == "Inbetween":

                # body_handle.color = viz_data.colors[1][person_idx]
                
           
                if t <4 or t > viz_data.timesteps - 5:
                    body_handle.color = gt_rgb_pickers[person_idx].value
                else:
                    # body_handle.color = viz_data.colors[1][person_idx]
                    s_idx = min(sample_num, len(viz_data.colors)-1)
                    body_handle.color = viz_data.colors[s_idx][person_idx]

                    
            elif color_mode == "Trajectory (idx)":
                # Use manually selected color from SLAHMR_COLORS
                slider_key = (sample_idx, person_idx)
                if slider_key in idx_color_sliders:
                    color_idx = int(idx_color_sliders[slider_key].value)
                    body_handle.color = SLAHMR_COLORS[color_idx]
                else:
                    # Fallback if slider doesn't exist
                    body_handle.color = viz_data.colors[1][person_idx]
            elif color_mode == "Trajectory (rgb)":
              
                s_idx = min(sample_num, len(viz_data.colors)-1)
                body_handle.color = viz_data.colors[s_idx][person_idx]
            else:
                assert False

    def draw_motion_trails(
        start_t, end_t, gt_shift, sample_shift, active_timesteps=None
    ):
        """Draw motion trails from start_t to end_t.

        Args:
            start_t: Starting timestep
            end_t: Ending timestep
            gt_shift: GT offset (scalar for Playback, vector for Multitimestep)
            sample_shift: Sample offset (scalar for Playback, vector for Multitimestep)
            active_timesteps: List of visible timestep values for interpolating frame indices (Multitimestep only)
        """
        nonlocal trail_handles

        # Remove old trails
        for handle in trail_handles.values():
            try:
                handle.remove()
            except:
                pass  # Handle already removed
        trail_handles.clear()

        for sample_num, sample_idx in enumerate(viz_data.sample_idx_list):
            # Check visibility based on GT/samples checkboxes
            if sample_idx == 0 and not show_gt_checkbox.value:
                continue
            if sample_idx > 0 and not show_samples_checkbox.value:
                continue

            if sample_num == 0:
                offset = np.array([0.0, 0.0, 0.0])
            elif sample_num == 1:
                # Handle both scalar (Playback) and vector (Multitimestep) modes
                offset = (
                    np.array([gt_shift, 0.0, 0.0])
                    if np.isscalar(gt_shift)
                    else gt_shift
                )
            else:
                # Handle both scalar (Playback) and vector (Multitimestep) modes
                if np.isscalar(gt_shift):
                    offset = np.array(
                        [gt_shift + sample_shift * float(sample_num - 1), 0.0, 0.0]
                    )
                else:
                    offset = gt_shift + sample_shift * float(sample_num - 1)

            for person_idx in range(viz_data.person_num):
                # Get base color for this person based on color mode
                color_mode = sample_color_mode_dropdown.value

                if color_mode == "Uniform":
                    base_color = viz_data.colors[1][person_idx]
                elif color_mode == "Inbetween":
                    base_color = viz_data.colors[1][person_idx]
                elif color_mode == "Trajectory (idx)":
                    slider_key = (sample_idx, person_idx)
                    if slider_key in idx_color_sliders:
                        color_idx = int(idx_color_sliders[slider_key].value)
                        base_color = SLAHMR_COLORS[color_idx]
                    else:
                        base_color = viz_data.colors[1][person_idx]
                elif color_mode == "Trajectory (rgb)":
                    # Use per-trajectory color
                    trajectory_key = (sample_idx, person_idx)
                    if trajectory_key in trajectory_rgb_pickers:
                        base_color = trajectory_rgb_pickers[trajectory_key].value
                    else:
                        base_color = viz_data.colors[1][person_idx]
                else:
                    assert False

                # Create positions for trail
                num_points = end_t - start_t + 1
                if num_points < 2:
                    continue

                points = []
                for t in range(start_t, end_t + 1):
                    if t < viz_data.timesteps:
                        T_world_root = viz_data.T_world_root_all[
                            sample_idx, t, person_idx
                        ]

                        # Calculate frame index for timestep offset
                        if active_timesteps is not None and len(active_timesteps) > 1:
                            # Multitimestep mode: interpolate based on active_timesteps
                            frame_idx = 0.0
                            for i in range(len(active_timesteps) - 1):
                                if active_timesteps[i] <= t <= active_timesteps[i + 1]:
                                    # Interpolate between frame i and frame i+1
                                    t_range = (
                                        active_timesteps[i + 1] - active_timesteps[i]
                                    )
                                    if t_range > 0:
                                        alpha = (t - active_timesteps[i]) / t_range
                                        frame_idx = i + alpha
                                    else:
                                        frame_idx = i
                                    break
                            else:
                                # t is at or after the last active timestep
                                if t >= active_timesteps[-1]:
                                    frame_idx = len(active_timesteps) - 1
                            timestep_offset = (
                                np.array(gui_timestep_offset_vector.value) * frame_idx
                            )
                        else:
                            # Playback mode: no timestep offset
                            timestep_offset = np.array([0.0, 0.0, 0.0])

                        pos = offset + T_world_root[4:7] + timestep_offset
                        points.append(pos)

                if len(points) < 2:
                    continue

                # Apply Gaussian smoothing to trajectory
                points_array = np.array(points)
                if points_array.shape[0] >= 3:
                    # Apply Gaussian filter to each dimension independently
                    points_smoothed = np.zeros_like(points_array)
                    for i in range(3):  # x, y, z
                        points_smoothed[:, i] = gaussian_filter1d(
                            points_array[:, i], sigma=2.0, mode="nearest"
                        )
                else:
                    points_smoothed = points_array

                # Create line segments
                starts = points_smoothed[:-1]
                ends = points_smoothed[1:]

                # Generate color gradient
                trail_start_bright = gui_trail_start_brightness.value
                trail_end_bright = gui_trail_end_brightness.value
                colors_list = get_color_gradient(
                    base_color, len(starts), trail_start_bright, trail_end_bright
                )

                # Convert colors to numpy array and reshape to match points
                # Each segment needs 2 colors (one for each vertex)
                colors = np.array(colors_list)  # Shape: (N, 3)
                colors_per_vertex = np.repeat(
                    colors[:, np.newaxis, :], 2, axis=1
                )  # Shape: (N, 2, 3)

                # Add line segments with thicker lines
                trail_key = f"trail_{sample_idx}_{person_idx}"
                trail_handles[trail_key] = server.scene.add_line_segments(
                    f"/trails/{trail_key}",
                    points=np.concatenate([starts, ends], axis=1).reshape(-1, 2, 3),
                    colors=colors_per_vertex,
                    line_width=3.0,
                )

    def do_update():
        if viz_data.curr_body_handles == [] or not is_update_ok:
            return

        # Get shift values based on mode
        if viz_data.mode == "Playback":
            gt_shift = gui_gt_shift_slider.value
            sample_shift = gui_shift_slider.value
            motion_trails_enabled = gui_motion_trails_playback.value
        else:  # Multitimestep
            gt_shift = np.array(gui_gt_shift_vector.value)
            sample_shift = np.array(gui_shift_vector.value)
            motion_trails_enabled = gui_motion_trails_multi.value

        if viz_data.mode == "Playback":
            # Playback mode: show single timestep
            t = gui_timestep.value

            for sample_num, sample_idx in enumerate(viz_data.sample_idx_list):
                # Check visibility based on GT/samples checkboxes
                if sample_idx == 0 and not show_gt_checkbox.value:
                    # Hide GT bodies
                    for person_idx in range(viz_data.person_num):
                        if sample_idx < len(viz_data.curr_body_handles):
                            body_handle = viz_data.curr_body_handles[sample_idx][
                                person_idx
                            ]
                            body_handle.visible = False
                    continue
                if sample_idx > 0 and not show_samples_checkbox.value:
                    # Hide sample bodies
                    for person_idx in range(viz_data.person_num):
                        if sample_idx < len(viz_data.curr_body_handles):
                            body_handle = viz_data.curr_body_handles[sample_idx][
                                person_idx
                            ]
                            body_handle.visible = False
                    continue

                if sample_num == 0:
                    offset = np.array([0.0, 0.0, 0.0])
                elif sample_num == 1:
                    offset = np.array([gt_shift, 0.0, 0.0])
                else:
                    offset = np.array(
                        [gt_shift + sample_shift * float(sample_num - 1), 0.0, 0.0]
                    )

                for person_idx in range(viz_data.person_num):
                    update_body_at_timestep(
                        sample_idx, sample_num, person_idx, t, offset
                    )

            # Update timestep frames
            for ii, timestep_frame in enumerate(timestep_handles):
                timestep_frame.visible = t == ii

            # Draw motion trails from 0 to current timestep
            if motion_trails_enabled:
                draw_motion_trails(0, t, gt_shift, sample_shift, active_timesteps=None)
            else:
                # Remove trails if disabled
                for handle in trail_handles.values():
                    try:
                        handle.remove()
                    except:
                        pass  # Handle already removed
                trail_handles.clear()

        else:  # Multitimestep mode
            nonlocal prev_multitimestep_state

            # Skip if multitimestep handles haven't been created yet
            if not viz_data.multi_timestep_handles or not multitimestep_sliders:
                return

            # Get current state
            current_selected_timesteps = tuple(
                slider.value for slider in multitimestep_sliders
            )
            current_checkbox_states = tuple(
                checkbox.value for checkbox in multitimestep_checkboxes
            )
            num_visible = len(multitimestep_sliders)

            # Include manual color slider values if in Manual mode
            if sample_color_mode_dropdown.value == "Trajectory (idx)":
                idx_colors = tuple(
                    slider.value for slider in idx_color_sliders.values()
                )
            else:
                idx_colors = ()

            # Include GT color values
            gt_colors = tuple(picker.value for picker in gt_rgb_pickers.values())

            # Include trajectory color values if in Trajectory mode
            if sample_color_mode_dropdown.value == "Trajectory (rgb)" or sample_color_mode_dropdown.value == "Inbetween":
                trajectory_colors = tuple(
                    picker.value for picker in trajectory_rgb_pickers.values()
                )
            else:
                trajectory_colors = ()

            current_state = (
                current_selected_timesteps,
                current_checkbox_states,
                tuple(gt_shift),
                tuple(sample_shift),
                motion_trails_enabled,
                show_gt_checkbox.value,
                show_samples_checkbox.value,
                gui_trail_start_brightness.value,
                gui_trail_end_brightness.value,
                gui_mesh_start_brightness.value,
                gui_mesh_end_brightness.value,
                sample_color_mode_dropdown.value,
                num_visible,  # Include number of sliders in state
                idx_colors,  # Include manual color slider values
                gt_colors,  # Include GT color values
                trajectory_colors,  # Include trajectory color values
                gui_timestep_offset_vector.value,  # Include timestep offset vector
            )

            # Only check state if we have valid handles with matching count
            if len(viz_data.multi_timestep_handles) == len(multitimestep_sliders):
                # Only update if something changed
                if prev_multitimestep_state == current_state:
                    return
            else:
                # Handles are being recreated, skip this frame to avoid flickering
                # The on_update callback will handle the recreation
                return

            # Update state
            prev_multitimestep_state = current_state

            # NOW hide all handles since we know we need to update
            # Always hide all regular body handles
            for sample_idx in range(len(viz_data.curr_body_handles)):
                for person_idx in range(viz_data.person_num):
                    if sample_idx < len(viz_data.curr_body_handles):
                        body_handle = viz_data.curr_body_handles[sample_idx][person_idx]
                        body_handle.visible = False

            # Always hide all multitimestep handles first
            for timestep_handles_list in viz_data.multi_timestep_handles:
                for sample_handles in timestep_handles_list:
                    for handle in sample_handles:
                        handle.visible = False

            # Update selected timesteps from sliders
            viz_data.selected_timesteps = list(current_selected_timesteps)

            # Show and update bodies at each selected timestep (only if checkbox is enabled)
            zero_pos = np.zeros(3, dtype=np.float32)
            active_timesteps = []  # Track which timesteps are visible for motion trails

            # num_visible already calculated above from len(multitimestep_sliders)
            # Use the minimum to be safe
            num_visible = min(
                num_visible,
                len(viz_data.selected_timesteps),
                len(multitimestep_checkboxes),
            )

            for timestep_idx in range(num_visible):
                # Safety checks
                if timestep_idx >= len(viz_data.multi_timestep_handles):
                    continue
                if timestep_idx >= len(viz_data.selected_timesteps):
                    continue
                if timestep_idx >= len(multitimestep_checkboxes):
                    continue

                t = viz_data.selected_timesteps[timestep_idx]

                # Check if this timestep is enabled via checkbox
                if not multitimestep_checkboxes[timestep_idx].value:
                    continue

                active_timesteps.append(t)

                # Iterate through visible samples only
                for sample_num, sample_idx in enumerate(viz_data.sample_idx_list):
                    # Check visibility based on GT/samples checkboxes
                    if sample_idx == 0 and not show_gt_checkbox.value:
                        continue
                    if sample_idx > 0 and not show_samples_checkbox.value:
                        continue

                    # Check if we have handles for this timestep
                    if timestep_idx >= len(viz_data.multi_timestep_handles):
                        continue

                    # Check if we have handles for this sample
                    if sample_idx >= len(viz_data.multi_timestep_handles[timestep_idx]):
                        continue

                    if sample_num == 0:
                        offset = np.array([0.0, 0.0, 0.0])
                    elif sample_num == 1:
                        offset = gt_shift
                    else:
                        offset = gt_shift + sample_shift * float(sample_num - 1)

                    for person_idx in range(viz_data.person_num):
                        # Check bounds before accessing
                        if sample_idx >= len(
                            viz_data.multi_timestep_handles[timestep_idx]
                        ):
                            continue
                        if person_idx >= len(
                            viz_data.multi_timestep_handles[timestep_idx][sample_idx]
                        ):
                            continue

                        body_handle = viz_data.multi_timestep_handles[timestep_idx][
                            sample_idx
                        ][person_idx]
                        body_handle.visible = True

                        if viz_data.timesteps <= t:
                            continue

                        # Update pose
                        T_world_root = viz_data.T_world_root_all[
                            sample_idx, t, person_idx
                        ]
                        body_handle.bones[0].position = zero_pos
                        body_handle.bones[0].wxyz = T_world_root[:4]
                        Ts_world_joint = viz_data.Ts_world_joint_all[
                            sample_idx, t, person_idx
                        ]
                        for b, bone_handle in enumerate(body_handle.bones[1:]):
                            bone_transform = Ts_world_joint[b]
                            bone_handle.position = bone_transform[4:7]
                            bone_handle.wxyz = bone_transform[:4]

                        # Calculate position with timestep offset
                        timestep_offset = (
                            np.array(gui_timestep_offset_vector.value) * timestep_idx
                        )
                        body_handle.position = (
                            offset + T_world_root[4:7] + timestep_offset
                        )

                        # Update color with gradient based on timestep position
                        # Get color mode first (needed for lightness gradient logic later)
                        color_mode = sample_color_mode_dropdown.value
                        print(f"color_mode: {color_mode}")

                        # GT always uses GT color from RGB picker
                        apply_lightness = True
                        if color_mode == "Inbetween":
                            # Use per-trajectory color with lightness gradient based on timestep
                            trajectory_key = (sample_idx, person_idx)
                            base_color = inbetween_rgb_pickers[trajectory_key].value
                            # Apply lightness gradient only to samples (not GT) and not first timestep
                            if sample_idx == 0 or t <4 or t > viz_data.timesteps - 4:
                                # GT or first timestep: use base color without lightness
                                body_handle.color = base_color
                            else:
                                # Samples at later timesteps: apply lightness gradient
                                # Calculate progress based on timestep
                                if viz_data.timesteps > 1:
                                    timestep_progress = t / (viz_data.timesteps - 1)
                                else:
                                    timestep_progress = 0
                                # Apply lightness gradient
                                mesh_start_bright = gui_mesh_start_brightness.value
                                mesh_end_bright = gui_mesh_end_brightness.value
                                lightness = (
                                    mesh_start_bright
                                    + (mesh_end_bright - mesh_start_bright)
                                    * timestep_progress
                                )
                                gradient_colors = get_color_gradient(
                                    base_color, 1, lightness, lightness
                                )
                                body_handle.color = gradient_colors[0]
                        elif sample_idx == 0:
                            if person_idx in gt_rgb_pickers:
                                base_color = gt_rgb_pickers[person_idx].value
                            else:
                                base_color = viz_data.colors[0][person_idx]
                            apply_lightness = False
                        elif t < viz_data.context_timesteps:
                            # Samples in context period: use GT color from RGB picker
                            if person_idx in gt_rgb_pickers:
                                base_color = gt_rgb_pickers[person_idx].value
                            else:
                                base_color = viz_data.colors[0][person_idx]
                            apply_lightness = False
                        elif color_mode == "Uniform":
                            # All samples use first sample color
                            base_color = viz_data.colors[1][person_idx]
                        elif color_mode == "Trajectory (idx)":
                            # Use manually selected color from SLAHMR_COLORS
                            slider_key = (sample_idx, person_idx)
                            if slider_key in idx_color_sliders:
                                color_idx = int(idx_color_sliders[slider_key].value)
                                base_color = SLAHMR_COLORS[color_idx]
                            else:
                                # Fallback if slider doesn't exist
                                base_color = viz_data.colors[1][person_idx]
                        elif color_mode == "Trajectory (rgb)":
                            # Use per-trajectory color with lightness gradient based on timestep
                            trajectory_key = (sample_idx, person_idx)
                            base_color = inbetween_rgb_pickers[trajectory_key].value
                            # Apply lightness gradient only to samples (not GT) and not first timestep
                            if sample_idx == 0 or t == 0:
                                # GT or first timestep: use base color without lightness
                                body_handle.color = base_color
                            else:
                                # Samples at later timesteps: apply lightness gradient
                                # Calculate progress based on timestep
                                if viz_data.timesteps > 1:
                                    timestep_progress = t / (viz_data.timesteps - 1)
                                else:
                                    timestep_progress = 0
                                # Apply lightness gradient
                                mesh_start_bright = gui_mesh_start_brightness.value
                                mesh_end_bright = gui_mesh_end_brightness.value
                                lightness = (
                                    mesh_start_bright
                                    + (mesh_end_bright - mesh_start_bright)
                                    * timestep_progress
                                )
                                gradient_colors = get_color_gradient(
                                    base_color, 1, lightness, lightness
                                )
                                body_handle.color = gradient_colors[0]
                        
                        else:
                          
                            assert False

                        # Apply lightness gradient based on actual timestep value
                        if not apply_lightness:
                            # GT or first timestep: use base color without lightness
                            body_handle.color = base_color
                        else:
                            # Samples at later timesteps: apply lightness gradient
                            mesh_start_bright = gui_mesh_start_brightness.value
                            mesh_end_bright = gui_mesh_end_brightness.value

                            # Calculate progress based on actual timestep value, not slider index
                            if viz_data.timesteps > 1:
                                timestep_progress = t / (viz_data.timesteps - 1)
                            else:
                                timestep_progress = 0

                            # Interpolate lightness based on timestep progress
                            lightness = (
                                mesh_start_bright
                                + (mesh_end_bright - mesh_start_bright)
                                * timestep_progress
                            )
                            gradient_colors = get_color_gradient(
                                base_color, 1, lightness, lightness
                            )
                            body_handle.color = gradient_colors[0]

            # Hide timestep frames
            for timestep_frame in timestep_handles:
                timestep_frame.visible = False

            # Draw motion trails between first and last active timesteps
            if motion_trails_enabled and len(active_timesteps) >= 2:
                start_t = min(active_timesteps)
                end_t = max(active_timesteps)
                draw_motion_trails(
                    start_t, end_t, gt_shift, sample_shift, active_timesteps
                )
            else:
                # Remove trails if disabled
                for handle in trail_handles.values():
                    try:
                        handle.remove()
                    except:
                        pass  # Handle already removed
                trail_handles.clear()

    get_viser_file = server.gui.add_button("Get .viser file")
    remove_old_bodies = server.gui.add_button("Remove old bodies")
    prev_time = time.time()
    handle = None

    @gui_next_frame.on_click
    def _(_):
        max_timestep = gui_timestep.max + 1
        gui_timestep.value = (gui_timestep.value + 1) % max_timestep

    @gui_prev_frame.on_click
    def _(_):
        max_timestep = gui_timestep.max + 1
        gui_timestep.value = (gui_timestep.value - 1) % max_timestep

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    @gui_framerate_options.on_click
    def _(_):
        gui_framerate.value = int(gui_framerate_options.value)

    # @gui_gt_shift_slider.on_update
    # def _(_):
    #     gui_gt_shift_slider.value = float(gui_gt_shift_slider.value)

    # @gui_shift_slider.on_update
    # def _(_):
    #     gui_shift_slider.value = float(gui_shift_slider.value)

    @refresh_file_list.on_click
    def _(_) -> None:
        subdir_dropdown.options = get_subdir_list()
        file_dropdown.options = get_file_list()

    @remove_old_bodies.on_click
    def _(_):
        remove_old_bodies.disabled = True
        gui_playing.value = False
        gui_timestep.disabled = False
        gui_next_frame.disabled = False
        gui_prev_frame.disabled = False

        # nonlocal old_body_handles
        nonlocal viz_data
        viz_data.remove_prev_data()
        remove_old_bodies.disabled = False

    @mode_dropdown.on_update
    def _(_):
        nonlocal viz_data
        nonlocal prev_multitimestep_state
        viz_data.mode = mode_dropdown.value

        # Reset state tracking when switching modes
        prev_multitimestep_state = None

        # Toggle folder visibility
        if viz_data.mode == "Playback":
            playback_folder.visible = True
            multitimestep_folder.visible = False

            # Hide all multitimestep handles
            for timestep_handles_list in viz_data.multi_timestep_handles:
                for sample_handles in timestep_handles_list:
                    for handle in sample_handles:
                        handle.visible = False

            # Make sure regular handles are visible
            for sample_idx in range(len(viz_data.curr_body_handles)):
                for person_idx in range(viz_data.person_num):
                    if sample_idx < len(viz_data.curr_body_handles):
                        body_handle = viz_data.curr_body_handles[sample_idx][person_idx]
                        body_handle.visible = sample_idx in viz_data.sample_idx_list

        else:  # Multitimestep
            playback_folder.visible = False
            multitimestep_folder.visible = True

            # Hide all regular body handles
            for sample_idx in range(len(viz_data.curr_body_handles)):
                for person_idx in range(viz_data.person_num):
                    if sample_idx < len(viz_data.curr_body_handles):
                        body_handle = viz_data.curr_body_handles[sample_idx][person_idx]
                        body_handle.visible = False

    @gui_num_visible_timesteps.on_update
    def _(_):
        nonlocal prev_multitimestep_state
        nonlocal viz_data

        # Force update on next frame
        prev_multitimestep_state = None

        # Hide all existing multitimestep handles before recreating
        for timestep_handles_list in viz_data.multi_timestep_handles:
            for sample_handles in timestep_handles_list:
                for handle in sample_handles:
                    handle.visible = False

        create_timestep_sliders()

    @sample_num_dropdown.on_update
    def _(_):
        nonlocal viz_data
        nonlocal server
        nonlocal prev_multitimestep_state

        if len(viz_data.sample_idx_list) - 1 == int(sample_num_dropdown.value):
            return

        sample_num = int(sample_num_dropdown.value)
        sample_num_ = min(sample_num, viz_data.all_sample_num)
        sample_idx_list = list(range(0, sample_num_ + 1))

        for sample_check_button in sample_check_buttons[:sample_num_]:
            sample_check_button.value = True
        for sample_check_button in sample_check_buttons[sample_num_:]:
            sample_check_button.value = False

        prev_multitimestep_state = None
        viz_data.update_sample_idx_list(sample_idx_list, server.scene)

    data_name = None
    subdir_name = None

    @subdir_dropdown.on_update
    def _(_):
        nonlocal subdir_name
        nonlocal data_name

        if subdir_name == subdir_dropdown.value:
            return
        subdir_name = subdir_dropdown.value

        if subdir_dropdown.value == "None":
            file_dropdown.options = get_file_list()
        else:
            file_dropdown.options = get_file_list(data_root_dir / subdir_dropdown.value)
            file_dropdown.value = "None"
            data_name = "None"

    @file_dropdown.on_update
    def _(_):
        nonlocal data_name
        nonlocal raw_data
        nonlocal prev_multitimestep_state

        if file_dropdown.value in [data_name, "None"]:
            return

        data_name = file_dropdown.value
        if subdir_dropdown.value == "None":
            raw_data = np.load(data_root_dir / file_dropdown.value)
        else:
            raw_data = np.load(
                data_root_dir / subdir_dropdown.value / file_dropdown.value
            )

        nonlocal is_update_ok
        is_update_ok = False
        gui_playing.value = False
        prev_multitimestep_state = None

        load_data()
        gui_timestep.value = 0
        gui_playing.value = True
        gui_timestep.disabled = True
        gui_next_frame.disabled = True
        gui_prev_frame.disabled = True

        is_update_ok = True

        nonlocal viz_data
        # viz_data.invisible_prev_data()
        # viz_data.invisible_data()

    @hand_pose_dropdown.on_update
    def _(_):
        nonlocal viz_data
        nonlocal raw_data
        nonlocal is_update_ok
        nonlocal prev_multitimestep_state

        if viz_data.hand_pose_type == hand_pose_dropdown.value:
            return

        viz_data.hand_pose_type = hand_pose_dropdown.value

        # Reload data if already loaded
        if raw_data is not None:
            is_update_ok = False
            gui_playing.value = False
            prev_multitimestep_state = None
            load_data()
            gui_timestep.value = 0
            gui_playing.value = True
            is_update_ok = True

    for sample_check_button in sample_check_buttons:

        @sample_check_button.on_update
        def _(_):
            sample_idx_list = [0]
            for sample_idx, sample_check_button in enumerate(sample_check_buttons):
                if sample_check_button.value:
                    sample_idx_list.append(sample_idx + 1)

            nonlocal viz_data
            nonlocal server
            nonlocal prev_multitimestep_state

            if viz_data.sample_idx_list == sample_idx_list:
                return

            prev_multitimestep_state = None
            viz_data.update_sample_idx_list(sample_idx_list, server.scene)
            sample_num_dropdown.value = str(len(sample_idx_list) - 1)

            # Update color controls for newly visible samples
            update_color_controls()

    prev_time = time.time()
    handle = None

    def loop_cb() -> int:
        start, end = gui_start_end.value
        duration = end - start

        if get_viser_file.value is False:
            nonlocal prev_time
            now = time.time()
            sleepdur = 1.0 / gui_framerate.value - (now - prev_time)
            inc = 1
            if sleepdur > 0.0:
                time.sleep(sleepdur)
            elif sleepdur < 0.0:
                inc = np.ceil((now - prev_time) * gui_framerate.value)
                sleepdur_ = inc / gui_framerate.value - (now - prev_time)
                time.sleep(sleepdur_)
            prev_time = now
            if gui_playing.value:
                gui_timestep.value = (
                    gui_timestep.value + inc - start
                ) % duration + start
            do_update()
            return gui_timestep.value
        else:
            # Save trajectory.
            nonlocal handle
            if handle is None:
                handle = server._start_scene_recording()
                handle.set_loop_start()
                gui_timestep.value = start

            assert handle is not None
            handle.insert_sleep(1.0 / gui_framerate.value)
            gui_timestep.value = (gui_timestep.value + 1 - start) % duration + start

            if gui_timestep.value == start:
                get_viser_file.value = False
                server.send_file_download(
                    "recording.viser", content=handle.end_and_serialize()
                )
                handle = None

            do_update()

            return gui_timestep.value

    return loop_cb


def visualize(
    server: viser.ViserServer,
    data_root_dir: Path | None,
) -> None:
    assert data_root_dir is not None

    loop_cb = load_visualizer(server, data_root_dir)

    while True:
        loop_cb()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--port", type=int, default=8084)
    args.add_argument("--data_dir", type=str, default="./dfot_outputs")
    args = args.parse_args()

    server = viser.ViserServer(port=args.port)
    visualize(server, data_root_dir=Path(args.data_dir))
