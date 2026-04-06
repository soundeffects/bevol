/*! # Voxel Module
*/

/// Represents a single voxel in a [`Volume`](super::Volume). Implementors can store arbitrary
/// data per-voxel while satisfying the constraints required for async generation and meshing.
///
/// # Example
/// ```
/// # use bevox::{Voxel, StandardVoxel};
/// // `bevox::StandardVoxel` is a provided implementer of `Voxel`
/// let voxel = StandardVoxel::new(1, 1.0);
/// assert!(voxel.opaque());
/// ```
pub trait Voxel: Clone + Copy + Default + Send + Sync {
    /// Returns the density value of this voxel, normalized to the range `[-1.0, 1.0]`, where
    /// `-1.0` represents empty space, `1.0` represents solid matter. The boundary between empty
    /// space and solid matter is determined by the implmementation of [`Voxel::opaque`].
    ///
    /// # Example
    /// ```
    /// # use bevox::{Voxel, StandardVoxel};
    /// assert!(StandardVoxel::new(0, 1.0).density() > 0.0);
    /// ```
    fn density(&self) -> f32;

    /// Returns `true` if this voxel should be considered solid (opaque) for mesh generation.
    ///
    /// The exact threshold depends on the implementation. For [`StandardVoxel`], voxels
    /// with density above 0.0 (midpoint) are opaque.
    ///
    /// # Example
    /// ```
    /// # use bevox::{Voxel, StandardVoxel};
    /// assert!(StandardVoxel::new(1, 1.0).opaque());
    /// ```
    fn opaque(&self) -> bool;
}

/// A simple voxel type storing a material identifier and a density value, provided for ease of
/// use.
///
/// # Example
/// ```
/// # use bevox::{Voxel, StandardVoxel};
/// let example = StandardVoxel::new(0, 1.0);
/// assert!(example.material_id() == 0);
/// assert!(example.opaque());
/// ```
#[derive(Clone, Copy, Default)]
pub struct StandardVoxel {
    /// An identifier for the material type of this voxel.
    pub material_id: u8,
    /// The internal density storage, normalized to `[0, 255]`.
    pub density: u8,
}

impl StandardVoxel {
    /// Creates a new `StandardVoxel` with the given material ID and density. The material ID
    /// should be within the range `[0, 255]` and the density will be clamped to the range
    /// `[-1.0, 1.0]`. Density values above `0.0` will be considered opaque.
    ///
    /// # Example
    /// ```
    /// # use bevox::{Voxel, StandardVoxel};
    /// let example = StandardVoxel::new(0, 1.0);
    /// assert!(example.material_id() == 0);
    /// assert!(example.opaque());
    /// ```
    pub fn new(material_id: u8, density: f32) -> Self {
        Self {
            material_id,
            density: ((density.clamp(-1.0, 1.0) + 1.0) * (u8::MAX as f32) / 2.0) as u8,
        }
    }

    /// Retrieves the material id of this voxel, which is stored as a `u8` type.
    ///
    /// # Example
    /// ```
    /// # use bevox::StandardVoxel;
    /// assert!(StandardVoxel::new(0, 0.0).material_id() == 0);
    /// ```
    pub fn material_id(&self) -> u8 {
        self.material_id
    }
}

impl Voxel for StandardVoxel {
    fn density(&self) -> f32 {
        let density = self.density as f32;
        let max = u8::MAX as f32;
        (density / max) * 2.0 - 1.0
    }

    fn opaque(&self) -> bool {
        self.density > u8::MAX / 2
    }
}

#[cfg(test)]
mod tests {
    use super::{StandardVoxel, Voxel};

    #[test]
    fn test_standard_voxel_new() {
        let voxel = StandardVoxel::new(1, 0.0);
        assert_eq!(voxel.material_id, 1);
        assert_eq!(voxel.density, u8::MAX / 2);
    }

    #[test]
    fn test_standard_voxel_density() {
        let voxel_max = StandardVoxel::new(0, 1.0);
        assert!((voxel_max.density() - 1.0).abs() < 0.01);

        let voxel_min = StandardVoxel::new(0, -1.0);
        assert!((voxel_min.density() - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_standard_voxel_opaque() {
        let opaque_voxel = StandardVoxel::new(0, 1.0);
        assert!(opaque_voxel.opaque());

        let transparent_voxel = StandardVoxel::new(0, -1.0);
        assert!(!transparent_voxel.opaque());
    }
}
