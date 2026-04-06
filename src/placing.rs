/*! # Placing Module
*/

use super::{Cell, Voxel};
use bevy::prelude::Vec3;

pub fn boxy_placer<V: Voxel>(_cell: Cell<V>) -> Vec3 {
    Vec3::ONE * 0.5
}

pub fn smooth_placer<V: Voxel>(_cell: Cell<V>) -> Vec3 {
    todo!()
}

pub fn contour_placer<V: Voxel>(_cell: Cell<V>) -> Vec3 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::{boxy_placer, contour_placer, smooth_placer, Voxel};
    use crate::StandardVoxel;

    #[test]
    fn test_placer_boxy() {
        let mut corners = [StandardVoxel::default(); 8];
        corners[0] = StandardVoxel::new(1, 1.0);
        let pos = boxy_placer(corners);
        assert!(pos.x >= 0.0 && pos.x <= 0.5);
    }

    #[test]
    fn test_placer_smooth() {
        let corners = [StandardVoxel::default(); 8];
        let pos = smooth_placer(corners);
        assert!(pos.x.is_finite());
        assert!(pos.y.is_finite());
        assert!(pos.z.is_finite());
    }

    #[test]
    fn test_placer_contoured() {
        let mut corners = [StandardVoxel::default(); 8];
        corners[0] = StandardVoxel::new(1, 1.0);
        corners[7] = StandardVoxel::new(1, -1.0);
        let pos = contour_placer(corners);
        assert!(pos.x >= -0.25 && pos.x <= 0.25);
    }
}
