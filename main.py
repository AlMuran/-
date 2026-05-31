import matplotlib.pyplot as plt
import math

# ========== Базовые параметры ==========
L_default = 200.0
g_default = 9.8
omega_earth = 7.2921e-5
gamma = 0.0

# Начальные условия по умолчанию
x0_def = 20.0
y0_def = 0.0
vx0_def = 0.0
vy0_def = 0.4

t_max = 5000.0      # с
dt = 0.1

cities = [
    ("Тромсё", 69.6),
    ("Кито", -0.2),
    ("Кейптаун", -33.9)
]

# ========== Функция расчёта траектории ==========
def compute_trajectory(phi_deg, L, g, method='rk4',
                       x0=x0_def, y0=y0_def, vx0=vx0_def, vy0=vy0_def):
    phi = math.radians(phi_deg)
    omega0 = math.sqrt(g / L)
    omega = omega_earth * math.sin(phi)
    Omega = omega / omega0
    beta = gamma / omega0

    xi0 = x0 / L
    eta0 = y0 / L
    u0 = vx0 / (L * omega0)
    v0 = vy0 / (L * omega0)

    h = dt * omega0
    tau_max = t_max * omega0

    def rhs(tau, w):
        xi, eta, u, v = w
        du = 2*Omega*v + (Omega*Omega - 1)*xi - beta*u
        dv = -2*Omega*u + (Omega*Omega - 1)*eta - beta*v
        return [u, v, du, dv]

    tau = 0.0
    w = [xi0, eta0, u0, v0]
    xs = [xi0 * L]
    ys = [eta0 * L]
    times = [0.0]

    while tau < tau_max - 1e-8:
        if method == 'rk4':
            k1 = rhs(tau, w)
            k2 = rhs(tau + h/2, [w[i] + h/2*k1[i] for i in range(4)])
            k3 = rhs(tau + h/2, [w[i] + h/2*k2[i] for i in range(4)])
            k4 = rhs(tau + h,   [w[i] + h*k3[i] for i in range(4)])
            for i in range(4):
                w[i] += h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        else:  # euler
            f = rhs(tau, w)
            for i in range(4):
                w[i] += h * f[i]
        tau += h
        times.append(tau / omega0)
        xs.append(w[0] * L)
        ys.append(w[1] * L)
    return xs, ys, times

# ========== 1. RK4 для трёх городов ==========
data_rk4 = []
for name, phi in cities:
    xv, yv, _ = compute_trajectory(phi, L_default, g_default, 'rk4')
    data_rk4.append((name, phi, xv, yv))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Метод Рунге-Кутты 4-го порядка\nL={L_default} м, g={g_default} м/с², t_max={t_max/3600:.1f} ч', fontsize=12)
for i, (name, phi, xv, yv) in enumerate(data_rk4):
    ax = axes[i]
    ax.plot(xv, yv, 'b-', lw=0.6)
    ax.set_title(f'{name}, широта {phi:.1f}°')
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-x0_def*1.2, x0_def*1.2)
    ax.set_ylim(-x0_def*1.2, x0_def*1.2)
plt.tight_layout()
plt.savefig('foucault_rk4.png', dpi=200)
plt.close()

# ========== 2. Эйлер для трёх городов ==========
data_euler = []
for name, phi in cities:
    xv, yv, _ = compute_trajectory(phi, L_default, g_default, 'euler')
    data_euler.append((name, phi, xv, yv))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Метод Эйлера\nL={L_default} м, g={g_default} м/с², t_max={t_max/3600:.1f} ч', fontsize=12)
for i, (name, phi, xv, yv) in enumerate(data_euler):
    ax = axes[i]
    ax.plot(xv, yv, 'r-', lw=0.6)
    ax.set_title(f'{name}, широта {phi:.1f}°')
    ax.set_xlabel('x, м'); ax.set_ylabel('y, м')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-x0_def*1.2, x0_def*1.2)
    ax.set_ylim(-x0_def*1.2, x0_def*1.2)
plt.tight_layout()
plt.savefig('foucault_euler.png', dpi=200)
plt.close()

# ========== 3. Сравнение методов для Тромсё ==========
tr_rk4 = data_rk4[0]
tr_euler = data_euler[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(tr_rk4[2], tr_rk4[3], 'b-', lw=0.8)
ax1.set_title('RK4 – гладкая розетка')
ax1.set_xlabel('x, м'); ax1.set_ylabel('y, м')
ax1.grid(True); ax1.set_aspect('equal')
ax1.set_xlim(-x0_def*1.2, x0_def*1.2)
ax1.set_ylim(-x0_def*1.2, x0_def*1.2)

ax2.plot(tr_euler[2], tr_euler[3], 'r-', lw=0.8)
ax2.set_title('Эйлер – накопление ошибки')
ax2.set_xlabel('x, м'); ax2.set_ylabel('y, м')
ax2.grid(True); ax2.set_aspect('equal')
ax2.set_xlim(-x0_def*1.2, x0_def*1.2)
ax2.set_ylim(-x0_def*1.2, x0_def*1.2)
plt.suptitle('Сравнение методов (Тромсё, φ=69.6°)')
plt.tight_layout()
plt.savefig('rk4_vs_euler.png', dpi=200)
plt.close()

# ========== 4. Влияние L (g фиксировано) ==========
g_fixed = 9.8
L_vals = [50, 100, 200, 400]
fig, axes = plt.subplots(1, len(L_vals), figsize=(16, 4))
fig.suptitle(f'Влияние L (g={g_fixed} м/с², RK4, Тромсё)', fontsize=12)
for i, L_val in enumerate(L_vals):
    xv, yv, _ = compute_trajectory(69.6, L_val, g_fixed, 'rk4')
    axes[i].plot(xv, yv, 'b-', lw=0.6)
    axes[i].set_title(f'L = {L_val} м')
    axes[i].set_xlabel('x, м'); axes[i].set_ylabel('y, м')
    axes[i].grid(True); axes[i].set_aspect('equal')
    axes[i].set_xlim(-x0_def*1.2, x0_def*1.2)
    axes[i].set_ylim(-x0_def*1.2, x0_def*1.2)
plt.tight_layout()
plt.savefig('compare_L.png', dpi=200)
plt.close()

# ========== 5. Влияние g (L фиксировано) ==========
L_fixed = 200
g_vals = [5, 9.8, 15]
fig, axes = plt.subplots(1, len(g_vals), figsize=(12, 4))
fig.suptitle(f'Влияние g (L={L_fixed} м, RK4, Тромсё)', fontsize=12)
for i, g_val in enumerate(g_vals):
    xv, yv, _ = compute_trajectory(69.6, L_fixed, g_val, 'rk4')
    axes[i].plot(xv, yv, 'b-', lw=0.6)
    axes[i].set_title(f'g = {g_val} м/с²')
    axes[i].set_xlabel('x, м'); axes[i].set_ylabel('y, м')
    axes[i].grid(True); axes[i].set_aspect('equal')
    axes[i].set_xlim(-x0_def*1.2, x0_def*1.2)
    axes[i].set_ylim(-x0_def*1.2, x0_def*1.2)
plt.tight_layout()
plt.savefig('compare_g.png', dpi=200)
plt.close()

# ========== 6. Начальные условия ==========
# Толчок из равновесия
xv_push, yv_push, _ = compute_trajectory(69.6, L_default, g_default, 'rk4',
                                         x0=0.0, y0=0.0, vx0=0.0, vy0=0.4)
# Отпускание из крайнего положения
xv_rel, yv_rel, _ = compute_trajectory(69.6, L_default, g_default, 'rk4',
                                       x0=20.0, y0=0.0, vx0=0.0, vy0=0.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(xv_push, yv_push, 'b-', lw=0.8)
ax1.set_title('Толчок из равновесия (x0=0, vy0=0.4)')
ax1.set_xlabel('x, м'); ax1.set_ylabel('y, м')
ax1.grid(True); ax1.set_aspect('equal')
ax1.set_xlim(-25, 25); ax1.set_ylim(-25, 25)

ax2.plot(xv_rel, yv_rel, 'b-', lw=0.8)
ax2.set_title('Отпускание из крайнего положения (x0=20, v0=0)')
ax2.set_xlabel('x, м'); ax2.set_ylabel('y, м')
ax2.grid(True); ax2.set_aspect('equal')
ax2.set_xlim(-25, 25); ax2.set_ylim(-25, 25)
plt.suptitle('Влияние начальных условий (RK4, Тромсё)')
plt.tight_layout()
plt.savefig('initial_conditions.png', dpi=200)
plt.close()

print("✅ Все графики сохранены для отчёта.")