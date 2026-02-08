"""
Attitude dynamics and control system for spacecraft.

Includes:
- Quaternion-based attitude representation
- Attitude propagation
- Control modes (nadir pointing, sun pointing, inertial)
- Reaction wheels, magnetorquers, thrusters
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum
from space_sim.core.frames import Vector3


class AttitudeMode(Enum):
    """Spacecraft attitude control modes."""
    INERTIAL = "inertial"  # Fixed inertial orientation
    NADIR_POINTING = "nadir"  # Earth-pointing
    SUN_POINTING = "sun"  # Sun-tracking
    VELOCITY_POINTING = "velocity"  # RAM direction
    THREE_AXIS_STABILIZED = "stabilized"  # Active stabilization


@dataclass
class Quaternion:
    """
    Unit quaternion for attitude representation.
    
    Convention: q = [q0, q1, q2, q3] = [scalar, vector]
    where q0 is the scalar part and (q1, q2, q3) is the vector part.
    """
    q0: float  # Scalar component
    q1: float  # Vector x component
    q2: float  # Vector y component
    q3: float  # Vector z component
    
    @staticmethod
    def identity() -> Quaternion:
        """Return identity quaternion (no rotation)."""
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    
    @staticmethod
    def from_axis_angle(axis: Vector3, angle_rad: float) -> Quaternion:
        """
        Create quaternion from axis-angle representation.
        
        Args:
            axis: Rotation axis (will be normalized)
            angle_rad: Rotation angle in radians
            
        Returns:
            Unit quaternion
        """
        # Normalize axis
        ax, ay, az = axis
        mag = math.sqrt(ax*ax + ay*ay + az*az)
        if mag < 1e-10:
            return Quaternion.identity()
        
        ax, ay, az = ax/mag, ay/mag, az/mag
        
        half_angle = angle_rad / 2.0
        sin_half = math.sin(half_angle)
        
        return Quaternion(
            math.cos(half_angle),
            ax * sin_half,
            ay * sin_half,
            az * sin_half
        )
    
    def normalize(self) -> Quaternion:
        """Return normalized quaternion."""
        mag = math.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
        if mag < 1e-10:
            return Quaternion.identity()
        return Quaternion(self.q0/mag, self.q1/mag, self.q2/mag, self.q3/mag)
    
    def conjugate(self) -> Quaternion:
        """Return conjugate quaternion (inverse rotation for unit quaternions)."""
        return Quaternion(self.q0, -self.q1, -self.q2, -self.q3)
    
    def multiply(self, other: Quaternion) -> Quaternion:
        """Quaternion multiplication (rotation composition)."""
        q0 = self.q0 * other.q0 - self.q1 * other.q1 - self.q2 * other.q2 - self.q3 * other.q3
        q1 = self.q0 * other.q1 + self.q1 * other.q0 + self.q2 * other.q3 - self.q3 * other.q2
        q2 = self.q0 * other.q2 - self.q1 * other.q3 + self.q2 * other.q0 + self.q3 * other.q1
        q3 = self.q0 * other.q3 + self.q1 * other.q2 - self.q2 * other.q1 + self.q3 * other.q0
        return Quaternion(q0, q1, q2, q3)
    
    def rotate_vector(self, v: Vector3) -> Vector3:
        """
        Rotate a vector using this quaternion.
        
        Args:
            v: Vector to rotate
            
        Returns:
            Rotated vector
        """
        # Convert vector to quaternion
        v_quat = Quaternion(0.0, v[0], v[1], v[2])
        
        # v_rotated = q * v * q_conjugate
        result = self.multiply(v_quat).multiply(self.conjugate())
        
        return (result.q1, result.q2, result.q3)
    
    def to_euler_angles(self) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
        
        Returns:
            (roll, pitch, yaw) in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (self.q0 * self.q1 + self.q2 * self.q3)
        cosr_cosp = 1.0 - 2.0 * (self.q1**2 + self.q2**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (self.q0 * self.q2 - self.q3 * self.q1)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (self.q0 * self.q3 + self.q1 * self.q2)
        cosy_cosp = 1.0 - 2.0 * (self.q2**2 + self.q3**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)


@dataclass
class AttitudeState:
    """
    Complete attitude state of a spacecraft.
    
    Attributes:
        quaternion: Attitude quaternion (body frame w.r.t. ECI)
        angular_velocity: Angular velocity vector in body frame (rad/s)
    """
    quaternion: Quaternion
    angular_velocity: Vector3  # rad/s in body frame
    
    @staticmethod
    def inertial() -> AttitudeState:
        """Return inertial attitude (no rotation, no angular velocity)."""
        return AttitudeState(
            quaternion=Quaternion.identity(),
            angular_velocity=(0.0, 0.0, 0.0)
        )


def propagate_attitude(state: AttitudeState, dt: float, 
                      torque: Optional[Vector3] = None,
                      inertia: Tuple[float, float, float] = (100.0, 100.0, 100.0)) -> AttitudeState:
    """
    Propagate attitude state forward in time.
    
    Uses Euler's equations for rigid body dynamics.
    
    Args:
        state: Current attitude state
        dt: Time step (seconds)
        torque: External torque in body frame (N·m)
        inertia: Moments of inertia [Ixx, Iyy, Izz] (kg·m²)
        
    Returns:
        New attitude state
    """
    q = state.quaternion
    omega = state.angular_velocity
    
    # Propagate quaternion using angular velocity
    # q_dot = 0.5 * omega_quat * q
    omega_quat = Quaternion(0.0, omega[0], omega[1], omega[2])
    q_dot_temp = omega_quat.multiply(q)
    q_dot = (0.5 * q_dot_temp.q0, 0.5 * q_dot_temp.q1, 
             0.5 * q_dot_temp.q2, 0.5 * q_dot_temp.q3)
    
    new_q = Quaternion(
        q.q0 + q_dot[0] * dt,
        q.q1 + q_dot[1] * dt,
        q.q2 + q_dot[2] * dt,
        q.q3 + q_dot[3] * dt
    ).normalize()
    
    # Propagate angular velocity using Euler's equations
    # I * omega_dot = torque - omega × (I * omega)
    Ix, Iy, Iz = inertia
    wx, wy, wz = omega
    
    if torque:
        tx, ty, tz = torque
    else:
        tx, ty, tz = 0.0, 0.0, 0.0
    
    # Euler's equations
    omega_dot_x = (tx + (Iy - Iz) * wy * wz) / Ix
    omega_dot_y = (ty + (Iz - Ix) * wz * wx) / Iy
    omega_dot_z = (tz + (Ix - Iy) * wx * wy) / Iz
    
    new_omega = (
        wx + omega_dot_x * dt,
        wy + omega_dot_y * dt,
        wz + omega_dot_z * dt
    )
    
    return AttitudeState(quaternion=new_q, angular_velocity=new_omega)


def compute_nadir_pointing_quaternion(r_eci: Vector3, v_eci: Vector3) -> Quaternion:
    """
    Compute quaternion for nadir (Earth) pointing attitude.
    
    Convention: Body +Z axis points to Earth center (nadir),
                Body +X axis points in velocity direction (RAM),
                Body +Y axis completes right-handed frame.
    
    Args:
        r_eci: Position vector in ECI (km)
        v_eci: Velocity vector in ECI (km/s)
        
    Returns:
        Quaternion for nadir pointing attitude
    """
    # Normalize position (points to nadir)
    r_mag = math.sqrt(r_eci[0]**2 + r_eci[1]**2 + r_eci[2]**2)
    if r_mag < 1e-10:
        return Quaternion.identity()
    
    z_body = (-r_eci[0]/r_mag, -r_eci[1]/r_mag, -r_eci[2]/r_mag)  # Nadir direction
    
    # Velocity direction (approximate RAM)
    v_mag = math.sqrt(v_eci[0]**2 + v_eci[1]**2 + v_eci[2]**2)
    if v_mag < 1e-10:
        return Quaternion.identity()
    
    x_body_approx = (v_eci[0]/v_mag, v_eci[1]/v_mag, v_eci[2]/v_mag)
    
    # Y axis = Z × X (right-handed)
    y_body = (
        z_body[1] * x_body_approx[2] - z_body[2] * x_body_approx[1],
        z_body[2] * x_body_approx[0] - z_body[0] * x_body_approx[2],
        z_body[0] * x_body_approx[1] - z_body[1] * x_body_approx[0]
    )
    
    y_mag = math.sqrt(y_body[0]**2 + y_body[1]**2 + y_body[2]**2)
    if y_mag < 1e-10:
        return Quaternion.identity()
    
    y_body = (y_body[0]/y_mag, y_body[1]/y_mag, y_body[2]/y_mag)
    
    # Recompute X = Y × Z for orthogonality
    x_body = (
        y_body[1] * z_body[2] - y_body[2] * z_body[1],
        y_body[2] * z_body[0] - y_body[0] * z_body[2],
        y_body[0] * z_body[1] - y_body[1] * z_body[0]
    )
    
    # Convert rotation matrix to quaternion
    # DCM = [x_body; y_body; z_body] (row vectors)
    trace = x_body[0] + y_body[1] + z_body[2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (y_body[2] - z_body[1]) * s
        q2 = (z_body[0] - x_body[2]) * s
        q3 = (x_body[1] - y_body[0]) * s
    else:
        if x_body[0] > y_body[1] and x_body[0] > z_body[2]:
            s = 2.0 * math.sqrt(1.0 + x_body[0] - y_body[1] - z_body[2])
            q0 = (y_body[2] - z_body[1]) / s
            q1 = 0.25 * s
            q2 = (y_body[0] + x_body[1]) / s
            q3 = (z_body[0] + x_body[2]) / s
        elif y_body[1] > z_body[2]:
            s = 2.0 * math.sqrt(1.0 + y_body[1] - x_body[0] - z_body[2])
            q0 = (z_body[0] - x_body[2]) / s
            q1 = (y_body[0] + x_body[1]) / s
            q2 = 0.25 * s
            q3 = (z_body[1] + y_body[2]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + z_body[2] - x_body[0] - y_body[1])
            q0 = (x_body[1] - y_body[0]) / s
            q1 = (z_body[0] + x_body[2]) / s
            q2 = (z_body[1] + y_body[2]) / s
            q3 = 0.25 * s
    
    return Quaternion(q0, q1, q2, q3).normalize()


@dataclass
class AttitudeController:
    """
    Simple PD controller for attitude control.
    """
    kp: float = 0.1  # Proportional gain
    kd: float = 0.5  # Derivative gain
    
    def compute_control_torque(self, 
                              current_attitude: AttitudeState,
                              desired_quaternion: Quaternion) -> Vector3:
        """
        Compute control torque to achieve desired attitude.
        
        Args:
            current_attitude: Current attitude state
            desired_quaternion: Desired attitude quaternion
            
        Returns:
            Control torque in body frame (N·m)
        """
        # Error quaternion: q_error = q_desired * q_current^conjugate
        q_error = desired_quaternion.multiply(current_attitude.quaternion.conjugate())
        
        # Proportional term (use vector part of error quaternion)
        error_vec = (q_error.q1, q_error.q2, q_error.q3)
        
        # Derivative term (current angular velocity)
        omega = current_attitude.angular_velocity
        
        # PD control law
        torque_x = self.kp * error_vec[0] - self.kd * omega[0]
        torque_y = self.kp * error_vec[1] - self.kd * omega[1]
        torque_z = self.kp * error_vec[2] - self.kd * omega[2]
        
        return (torque_x, torque_y, torque_z)
