from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FeatureRegistry:
    # Built-in road features computed from geometry or copied from extra road fields.
    road_features: Dict[str, bool] = field(default_factory=lambda: {
        'length': True,
        'bearing': True,
        'dx': True,
        'dy': True,
        'abs_dx': True,
        'abs_dy': True,
        'norm_dx': True,
        'norm_dy': True,
        'mid_x': True,
        'mid_y': True,
        'curvature_proxy': True,
        'bbox_width': True,
        'bbox_height': True,
        'slope': False,
        'nearest_point_dist': False,
    })

    # Built-in node features. Extra node fields can be manually edited in prep_*_nodes
    # and then read during export/predict by adding field names to extra_node_fields.
    node_features: Dict[str, bool] = field(default_factory=lambda: {
        'x': True,
        'y': True,
        'degree': True,
        'dead_end': True,
        'junction': True,
        'component_size': True,
        'incident_len_mean': True,
        'incident_len_max': True,
        'incident_len_std': True,
        'dominant_angle': True,
        'local_deadend_ratio': True,
        'local_junction_ratio': True,
        'nearest_node_dist': False,
        'elevation': False,
    })

    # Built-in candidate / bridge features.
    candidate_features: Dict[str, bool] = field(default_factory=lambda: {
        'gap_len': True,
        'gap_bearing': True,
        'angle_u': True,
        'angle_v': True,
        'angle_to_gap_u': True,
        'angle_to_gap_v': True,
        'angle_diff': True,
        'comp_same': True,
        'deg_u': True,
        'deg_v': True,
        'dead_u': True,
        'dead_v': True,
        'comp_size_u': True,
        'comp_size_v': True,
        'comp_size_ratio': True,
        'degree_sum': True,
        'deadend_pair': True,
        'incident_len_mean_u': True,
        'incident_len_mean_v': True,
        'incident_len_std_u': True,
        'incident_len_std_v': True,
        'dominant_angle_diff': True,
        'midpoint_density': True,
        'detour_ratio_proxy': True,
        'crossing_proxy': True,
        'slope_along_gap': False,
        'mask_support': False,
    })

    # Extra feature fields you want to carry from GIS manually or external preprocessing.
    # For roads: set the raw field names that exist in prep_*_road.
    extra_road_fields: List[str] = field(default_factory=list)
    # For nodes: set names like ['manual_score', 'dem_z'] and fill them in prep_*_nodes.
    # The scripts will read them as NF_manual_score, NF_dem_z.
    extra_node_fields: List[str] = field(default_factory=list)
    # For candidates: set names like ['cnn_support', 'prob_hint'] and fill them in prep_*_candidates.
    # The scripts will read/write them as CF_cnn_support, CF_prob_hint.
    extra_candidate_fields: List[str] = field(default_factory=list)

    def enabled_names(self, group: str) -> List[str]:
        d = getattr(self, group)
        return [k for k, v in d.items() if v]

    def all_candidate_names(self) -> List[str]:
        return self.enabled_names('candidate_features') + list(self.extra_candidate_fields)

    def all_node_names(self) -> List[str]:
        return self.enabled_names('node_features') + list(self.extra_node_fields)

    def all_road_names(self) -> List[str]:
        return self.enabled_names('road_features') + list(self.extra_road_fields)


DEFAULT_REGISTRY = FeatureRegistry()
