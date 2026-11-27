//! An orbit camera around a model: yaw and pitch about a target, a distance, and a pan.

use glam::{Mat4, Vec3};

/// Where the camera looks from and to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orbit {
    pub target: Vec3,
    /// Radians around +Y, 0 looking down -Z.
    pub yaw: f32,
    /// Radians above the horizon.
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view, radians.
    pub fov: f32,
    /// The model's radius, for near and far planes and zoom limits.
    pub radius: f32,
}

impl Orbit {
    /// A three-quarter view of a sphere at `center` of `radius`, the whole of it in frame.
    #[must_use]
    pub fn framing(center: Vec3, radius: f32) -> Self {
        let radius = radius.max(1e-4);
        let fov = 40f32.to_radians();
        Self {
            target: center,
            yaw: 35f32.to_radians(),
            pitch: 18f32.to_radians(),
            distance: radius / (fov / 2.0).sin() * 1.2,
            fov,
            radius,
        }
    }

    /// The camera's position.
    #[must_use]
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(sy * cp, sp, cy * cp) * self.distance
    }

    /// Clip from world, for a viewport `aspect` wide to its height.
    #[must_use]
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance - self.radius * 4.0).max(self.distance * 0.002).max(1e-5);
        let far = self.distance + self.radius * 12.0;
        let proj = glam::camera::rh::proj::directx::perspective(self.fov, aspect.max(1e-3), near, far);
        proj * glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Turns the camera by a drag of `dx`, `dy` pixels.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.54, 1.54);
    }

    /// Moves the target across the view by a drag of `dx`, `dy` pixels in a viewport `height` tall.
    pub fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        let per_pixel = 2.0 * self.distance * (self.fov / 2.0).tan() / height.max(1.0);
        self.target += (-right * dx + up * dy) * per_pixel;
    }

    /// Moves in (`factor` below 1) or out.
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(self.radius * 0.05, self.radius * 60.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_sees_the_whole_sphere() {
        let o = Orbit::framing(Vec3::new(1.0, 2.0, 3.0), 2.0);
        let vp = o.view_proj(1.0);
        // Points on the sphere project inside the clip volume.
        for d in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let p = vp.project_point3(o.target + d * 2.0);
            assert!(
                p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z),
                "{d} -> {p}"
            );
        }
    }
}
