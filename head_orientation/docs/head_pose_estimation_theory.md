# Head Pose Estimation: Theory & Algorithm Design (v4 — Yaw-Decoupled)

## 1. Problem Statement & Reference Alignment

Given 2D facial landmarks from dlib's 68-point shape predictor, estimate the driver's head pose orientation with **complete decoupling between all three rotational axes**:

- **Pitch** (primary goal) — vertical rotation: looking up (+) or drooping down (-)
- **Roll** (secondary) — lateral tilt: ear-to-shoulder (+ tilt left, - tilt right)
- **Yaw** (secondary / verification) — lateral turn: looking left (+) or looking right (-)

Reference image: `head_orientation/input/head_orientation_reference.png`

### Sign Conventions (Aligned with Reference Diagram)

| Axis | Positive (+) | Zero (0°) | Negative (-) |
|------|--------------|-----------|--------------|
| **Pitch** | Looking **UP** | Forward | Looking **DOWN / Drooping** |
| **Roll** | Tilting **Image-Left** (Subject's right ear down) | Upright | Tilting **Image-Right** (Subject's left ear down) |
| **Yaw** | Turning **Image-Left** (Subject's right) | Forward | Turning **Image-Right** (Subject's left) |

---

## 2. Root Cause: Why Yaw Was Being Accounted as Pitch

In the previous revision, pitch was calculated as:

$$d_{\text{top}} = (P_{33} - E_{\text{mid}}) \cdot \vec{u}_y$$
$$\eta = \frac{d_{\text{top}}}{W_{\text{eyes}}}$$
$$\theta_{\text{pitch}} = -\arctan\left(\frac{\eta - \eta_0}{k_p}\right)$$

### The Geometric Flaw: Dividing a Vertical Vector by a Horizontal Vector

1. $d_{\text{top}}$ is a **vertical distance** (projected onto the vertical facial axis $\vec{u}_y$).
2. $W_{\text{eyes}} = \|P_{45} - P_{36}\|$ is a **horizontal distance** (the distance between the two eyes).
3. When the user turns their head in **yaw** by angle $\psi$:
   - In 3D, rotation is around the vertical axis. The vertical distance $d_{\text{top}}$ does **not** change.
   - However, the horizontal eye width foreshortens:
     $$W_{\text{eyes}}(\psi) = W_{\text{eyes}}^0 \cdot \cos(\psi)$$
4. When $W_{\text{eyes}}$ is in the denominator:
   $$\eta(\psi) = \frac{d_{\text{top}}}{W_{\text{eyes}}^0 \cdot \cos(\psi)} = \frac{\eta_0}{\cos(\psi)}$$
5. For a standard $30^\circ$ head turn ($\cos(30^\circ) \approx 0.866$):
   - $W_{\text{eyes}}$ shrinks by $13.4\%$.
   - $\eta$ increases by $+15.5\%$.
   - $\theta_{\text{pitch}}$ computed as $-15.4^\circ$!

> [!CAUTION]
> Dividing vertical displacement by horizontal eye width creates an unavoidable $\frac{1}{\cos(\psi)}$ error. Turning the head by $25^\circ\text{–}30^\circ$ in yaw falsely registered as a $-15^\circ$ downward head droop (triggering a false Fatigue Warning).

---

## 3. The Solution: Purely Vertical Facial Height Normalization

To eliminate yaw contamination completely, the scale normalizer must be **parallel to the rotation axis of yaw** (i.e., vertical).

### 3.1 The Invariance Theorem for 3D Rotations

For any 3D vector $\vec{v}$ and rotation matrix $R_{\vec{u}}(\psi)$ around axis $\vec{u}$:

$$(R_{\vec{u}}(\psi) \vec{v}) \cdot \vec{u} = \vec{v} \cdot \vec{u}$$

Since **yaw rotates around the vertical axis $\vec{u}_y$**, the projection of **any landmark vector onto $\vec{u}_y$ is mathematically invariant to yaw**.

### 3.2 Defining the Vertical Proportions

Using the face-aligned coordinate system:
- Midpoint of outer eye corners: $E_{\text{mid}} = \frac{P_{36} + P_{45}}{2}$
- Face vertical axis: $\vec{u}_y$ (perpendicular to eye line, pointing downward)
- Vertical distance from eye level to nose tip:
  $$d_{\text{top}} = (P_{33} - E_{\text{mid}}) \cdot \vec{u}_y$$
- Vertical distance from nose tip to chin:
  $$d_{\text{bottom}} = (P_8 - P_{33}) \cdot \vec{u}_y$$
- Total face vertical height:
  $$H_{\text{face}} = d_{\text{top}} + d_{\text{bottom}} = (P_8 - E_{\text{mid}}) \cdot \vec{u}_y$$

### 3.3 The Pure Vertical Ratio $\rho$

We define the vertical nose position ratio as:

$$\rho = \frac{d_{\text{top}}}{H_{\text{face}}} = \frac{d_{\text{top}}}{d_{\text{top}} + d_{\text{bottom}}}$$

**Why this ratio is completely invariant to Yaw and Scale**:
1. **Scale-invariant**: When the user moves closer/further from camera, both $d_{\text{top}}$ and $H_{\text{face}}$ scale by $s$. The ratio $\frac{s \cdot d_{\text{top}}}{s \cdot H_{\text{face}}} = \rho$ is unchanged.
2. **Yaw-invariant**: Both $d_{\text{top}}$ and $H_{\text{face}}$ are vertical projections along $\vec{u}_y$. Neither contracts by $\cos(\psi)$. The ratio is completely unaffected by yaw!
3. **Roll-invariant**: Both distances are projected onto $\vec{u}_y$, which rotates with head tilt.

---

## 4. Pitch Mathematical Formulation

Under a pitch rotation $\phi$ (rotation around the horizontal transverse axis):

- Eye level $Y'_e = -Y_e \cos\phi$
- Chin $Y'_c = +Y_c \cos\phi$
- Nose tip (protrudes in depth by $Z_{\text{nose}} > 0$): $Y'_n = -Z_{\text{nose}} \sin\phi$

Projecting onto $\vec{u}_y$:
$$d_{\text{top}}(\phi) = Y'_n - Y'_e = Y_e \cos\phi - Z_{\text{nose}} \sin\phi$$
$$H_{\text{face}}(\phi) = Y'_c - Y'_e = (Y_e + Y_c) \cos\phi$$

Dividing to obtain the vertical ratio $\rho(\phi)$:

$$\rho(\phi) = \frac{Y_e \cos\phi - Z_{\text{nose}} \sin\phi}{(Y_e + Y_c) \cos\phi} = \frac{Y_e}{Y_e + Y_c} - \left(\frac{Z_{\text{nose}}}{Y_e + Y_c}\right) \tan\phi$$

Let:
- $\rho_0 = \frac{Y_e}{Y_e + Y_c}$ (baseline ratio at neutral pose, calibrated per user)
- $K_{\text{pitch}} = \frac{Z_{\text{nose}}}{Y_e + Y_c} \approx 0.18$ (anthropometric ratio of nose protrusion to face height)

Then:

$$\rho(\phi) = \rho_0 - K_{\text{pitch}} \tan\phi$$

Solving for pitch angle $\theta_{\text{pitch}}$:

$$\tan\phi = \frac{\rho_0 - \rho(\phi)}{K_{\text{pitch}}}$$

$$\theta_{\text{pitch}} = \arctan\left(\frac{\rho_0 - \rho}{K_{\text{pitch}}}\right) \times \frac{180}{\pi}$$

### Sign Verification
- **Looking UP ($\phi > 0$):** Nose tip moves up towards eyes $\implies d_{\text{top}} \downarrow \implies \rho < \rho_0 \implies \rho_0 - \rho > 0 \implies \theta_{\text{pitch}} > 0$ **(+)**
- **Looking DOWN ($\phi < 0$, droop):** Nose tip drops towards chin $\implies d_{\text{top}} \uparrow \implies \rho > \rho_0 \implies \rho_0 - \rho < 0 \implies \theta_{\text{pitch}} < 0$ **(-)**
- **Neutral ($\phi = 0$):** $\rho = \rho_0 \implies \theta_{\text{pitch}} = 0^\circ$

---

## 5. Roll & Yaw Formulations

### 5.1 Roll Estimation ($\theta_{\text{roll}}$)
From the eye line vector $\vec{v}_{\text{eyes}} = P_{45} - P_{36}$:

$$\theta_{\text{roll}} = \text{atan2}(v_{\text{eyes}, y}, \; v_{\text{eyes}, x}) \times \frac{180}{\pi}$$

- **Tilting Image-Left (+):** Right ear down $\implies \theta_{\text{roll}} > 0$
- **Tilting Image-Right (-):** Left ear down $\implies \theta_{\text{roll}} < 0$

### 5.2 Yaw Estimation ($\theta_{\text{yaw}}$)
From the horizontal asymmetry between the nose tip and outer eye corners:
- Distance to right eye corner: $d_R = \|P_{33} - P_{36}\|$
- Distance to left eye corner: $d_L = \|P_{33} - P_{45}\|$

$$\text{asym}_{\text{yaw}} = \frac{d_L - d_R}{d_L + d_R}$$

$$\theta_{\text{yaw}} = \arcsin\left(\text{clip}\left(\frac{\text{asym}_{\text{yaw}} - \text{asym}_0}{k_{\text{yaw}}},\; -1,\; 1\right)\right) \times \frac{180}{\pi}$$

where $k_{\text{yaw}} \approx 0.60$.
- **Turning Image-Left (+):** $d_L > d_R \implies \theta_{\text{yaw}} > 0$
- **Turning Image-Right (-):** $d_R > d_L \implies \theta_{\text{yaw}} < 0$

---

## 6. Decoupling Summary Matrix

| Axis | Primary Feature | Formula | Independence from other 2 axes |
|------|-----------------|---------|--------------------------------|
| **Pitch** | Vertical Ratio $\rho = \frac{d_{\text{top}}}{H_{\text{face}}}$ | $\arctan\left(\frac{\rho_0 - \rho}{K_{\text{pitch}}}\right)$ | **100% immune to Roll** (via $\vec{u}_y$) and **100% immune to Yaw** (pure vertical ratio) |
| **Roll** | Eye vector slope | $\text{atan2}(v_{\text{eyes}, y}, v_{\text{eyes}, x})$ | **100% immune to Pitch and Yaw** (measures 2D orientation of horizon) |
| **Yaw** | Bilateral Eye-Nose Asymmetry | $\arcsin\left(\frac{\Delta\text{asym}}{k_{\text{yaw}}}\right)$ | **Immune to Roll** (distances are rotation-invariant) and **Immune to Pitch** (symmetric along midline) |

---

## 7. Calibration & Fatigue Classification

### 7.1 Startup Calibration (60 frames / ~2 sec)
User looks straight at the camera:
- $\rho_0 = \text{median}(\rho)$
- $\theta_{\text{roll}, 0} = \text{median}(\theta_{\text{roll}})$
- $\text{asym}_0 = \text{median}(\text{asym}_{\text{yaw}})$

### 7.2 Fatigue Classification
- **NORMAL**: $\theta_{\text{pitch}} > -8^\circ$
- **MILD_DROOP**: $-8^\circ \ge \theta_{\text{pitch}} > -15^\circ$ sustained for $> 20$ frames
- **FATIGUE_WARNING**: $\theta_{\text{pitch}} \le -15^\circ$ sustained for $> 20$ frames

### 7.3 Yaw Gating (Distraction vs. Fatigue)
When $|\theta_{\text{yaw}}| > 20^\circ$, the driver is turning their head sideways (e.g. mirrors or shoulder check). The classifier marks the state as **LOOKING_AWAY** rather than false head drooping.
