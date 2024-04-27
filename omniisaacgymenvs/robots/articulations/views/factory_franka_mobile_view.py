
from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class MobileFrankaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "MobileFrankaView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._base = RigidPrimView(prim_paths_expr="/World/envs/.*/franka_mobile/darkoBASEONLY/darko_base_link", name="base_view", reset_xform_properties=False)
        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/franka_mobile/darkoBASEONLY/factory_franka_instanceable/panda_link7", name="hands_view", reset_xform_properties=False)
        self._lfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka_mobile/darkoBASEONLY/factory_franka_instanceable/panda_leftfinger", name="lfingers_view", reset_xform_properties=False)
        self._rfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka_mobile/darkoBASEONLY/factory_franka_instanceable/panda_rightfinger",  name="rfingers_view", reset_xform_properties=False)
        self._fingertip_centered = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka_mobile/darkoBASEONLY/factory_franka_instanceable//panda_fingertip_centered",
            name="fingertips_view",
            reset_xform_properties=False,
        )
        self._wheels = RigidPrimView(prim_paths_expr="/World/envs/.*/franka_mobile//darkoBASEONLY/wheel_*/omniwheel", name="wheels_view", reset_xform_properties=False,
                                     masses=omniwheels_mass,
                                     track_contact_forces=False, prepare_contact_sensors=False)
    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), 
                                 self.get_dof_index("panda_finger_joint2")]
        self._base_indices = [self.get_dof_index("wheel_LH_joint"), 
                              self.get_dof_index("wheel_RH_joint"), 
                              self.get_dof_index("wheel_LF_joint"),
                              self.get_dof_index("wheel_RF_joint")]
        
    @property
    def gripper_indices(self):
        return self._gripper_indices
    
