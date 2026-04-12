/*! # Placing Module

Provides vertex placement strategies for dual-contouring meshing. Each placer determines
where vertices are positioned within a cell to approximate the isosurface.
*/

use super::{Cell, Voxel};
use bevy::prelude::Vec3;

/// Places vertices at the cell center (0.5, 0.5, 0.5), producing box-like geometry.
/// This is the simplest placement strategy but produces blocky meshes.
///
/// # Example
/// ```
/// # use bevox::{boxy_placer, StandardVoxel, Cell};
/// # use bevy::prelude::Vec3;
/// let cell = Cell {
///     nx_ny_nz: StandardVoxel::default(),
///     nx_ny_pz: StandardVoxel::default(),
///     nx_py_nz: StandardVoxel::default(),
///     nx_py_pz: StandardVoxel::default(),
///     px_ny_nz: StandardVoxel::default(),
///     px_ny_pz: StandardVoxel::default(),
///     px_py_nz: StandardVoxel::default(),
///     px_py_pz: StandardVoxel::default(),
/// };
/// let pos = boxy_placer(cell);
/// assert_eq!(pos, Vec3::splat(0.5));
/// ```
pub fn boxy_placer<V: Voxel>(_cell: Cell<V>) -> Vec3 {
    Vec3::ONE * 0.5
}

/// Places vertices using linear interpolation based on voxel densities, producing
/// smoother geometry than [`boxy_placer`]. The vertex position is weighted toward
/// voxels with higher density values.
///
/// # Example
/// ```
/// # use bevox::{smooth_placer, StandardVoxel, Cell};
/// let mut cell = Cell {
///     nx_ny_nz: StandardVoxel::default(),
///     nx_ny_pz: StandardVoxel::new(0, 0.5),
///     nx_py_nz: StandardVoxel::default(),
///     nx_py_pz: StandardVoxel::default(),
///     px_ny_nz: StandardVoxel::default(),
///     px_ny_pz: StandardVoxel::default(),
///     px_py_nz: StandardVoxel::default(),
///     px_py_pz: StandardVoxel::default(),
/// };
/// let pos = smooth_placer(cell);
/// assert!(pos.x.is_finite());
/// ```
pub fn smooth_placer<V: Voxel>(cell: Cell<V>) -> Vec3 {
    let fields = cell.fields();
    let mut sum = Vec3::ZERO;
    let mut total_weight = 0.0;

    for (i, voxel) in fields.iter().enumerate() {
        let weight = voxel.density().max(0.0);
        let offset = Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32);
        sum += offset * weight;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        sum / total_weight
    } else {
        Vec3::splat(0.5)
    }
}

/// Places vertices using dual-contouring style positioning, which considers both
/// density and material boundaries for more accurate surface placement.
///
/// Currently uses a basic implementation that interpolates between opposing corners
/// based on the sign change in density.
pub fn contour_placer<V: Voxel>(cell: Cell<V>) -> Vec3 {
    let fields = cell.fields();
    let first_opaque = fields[0].opaque();

    let mut sum = Vec3::ZERO;
    let mut count = 0.0;

    for (i, voxel) in fields.iter().enumerate() {
        if voxel.opaque() != first_opaque {
            let offset = Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32);
            sum += offset;
            count += 1.0;
        }
    }

    if count > 0.0 {
        sum / count
    } else {
        Vec3::splat(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::{boxy_placer, contour_placer, smooth_placer, Cell};
    use crate::StandardVoxel;
    use bevy::prelude::Vec3;

    fn make_test_cell(solid_corner: usize, solid_density: f32) -> Cell<StandardVoxel> {
        let mut cell = Cell {
            nx_ny_nz: StandardVoxel::default(),
            nx_ny_pz: StandardVoxel::default(),
            nx_py_nz: StandardVoxel::default(),
            nx_py_pz: StandardVoxel::default(),
            px_ny_nz: StandardVoxel::default(),
            px_ny_pz: StandardVoxel::default(),
            px_py_nz: StandardVoxel::default(),
            px_py_pz: StandardVoxel::default(),
        };

        match solid_corner {
            0 => cell.nx_ny_nz = StandardVoxel::new(1, solid_density),
            1 => cell.nx_ny_pz = StandardVoxel::new(1, solid_density),
            2 => cell.nx_py_nz = StandardVoxel::new(1, solid_density),
            3 => cell.nx_py_pz = StandardVoxel::new(1, solid_density),
            4 => cell.px_ny_nz = StandardVoxel::new(1, solid_density),
            5 => cell.px_ny_pz = StandardVoxel::new(1, solid_density),
            6 => cell.px_py_nz = StandardVoxel::new(1, solid_density),
            7 => cell.px_py_pz = StandardVoxel::new(1, solid_density),
            _ => {}
        }
        cell
    }

    #[test]
    fn test_placer_boxy() {
        let cell = make_test_cell(0, 1.0);
        let pos = boxy_placer(cell);
        assert_eq!(pos, Vec3::splat(0.5));
    }

    #[test]
    fn test_placer_smooth() {
        let cell = Cell::<StandardVoxel>::default();
        let pos = smooth_placer(cell);
        assert!(pos.x.is_finite());
        assert!(pos.y.is_finite());
        assert!(pos.z.is_finite());
    }

    #[test]
    fn test_placer_smooth_with_density() {
        let mut cell = Cell {
            nx_ny_nz: StandardVoxel::new(0, 1.0),
            nx_ny_pz: StandardVoxel::new(0, 0.5),
            nx_py_nz: StandardVoxel::default(),
            nx_py_pz: StandardVoxel::default(),
            px_ny_nz: StandardVoxel::default(),
            px_ny_pz: StandardVoxel::default(),
            px_py_nz: StandardVoxel::default(),
            px_py_pz: StandardVoxel::default(),
        };
        let pos = smooth_placer(cell);
        assert!(pos.x > 0.0 && pos.x < 1.0);
    }

    #[test]
    fn test_placer_contoured() {
        let cell = make_test_cell(0, 1.0);
        let pos = contour_placer(cell);
        assert!(pos.x >= 0.0 && pos.x <= 1.0);
    }

    #[test]
    fn test_placer_contoured_mixed_opacity() {
        let mut cell = Cell {
            nx_ny_nz: StandardVoxel::new(1, 1.0),
            nx_ny_pz: StandardVoxel::default(),
            nx_py_nz: StandardVoxel::default(),
            nx_py_pz: StandardVoxel::default(),
            px_ny_nz: StandardVoxel::default(),
            px_ny_pz: StandardVoxel::default(),
            px_py_nz: StandardVoxel::default(),
            px_py_pz: StandardVoxel::default(),
        };
        let pos = contour_placer(cell);
        assert!(pos.x >= 0.0 && pos.x <= 1.0);
    }
}
