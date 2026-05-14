import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# ========== Параметры модели (без затухания) ==========
L = 200.0                # длина маятника, м
g = 9.8
omega_earth = 7.2921e-5  # рад/с
gamma = 0.0              # без трения – виден только эффект Фуко

# Начальные условия (создают эллиптическую розетку)
x0 = 20.0
y0 = 0.0
vx0 = 0.0
vy0 = 0.4                # поперечный толчок

t_max = 5000.0           # время симуляции, с (~1.4 часа)
dt = 0.1

# Города с очень разными широтами
cities = [
    ("Тромсё, Норвегия", 69.6),   # север, быстрый поворот против ч.с.
    ("Кито, Эквадор", -0.2),      # экватор, поворот почти нулевой
    ("Кейптаун, ЮАР", -33.9)      # юг, поворот по ч.с., медленнее
]

def compute_trajectory(phi_deg):
    phi = math.radians(phi_deg)
    omega0 = math.sqrt(g / L)
    omega = omega_earth * math.sin(phi)   # на юге отрицательное
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
    x_vals = [xi0 * L]
    y_vals = [eta0 * L]
    times = [0.0]

    while tau < tau_max - 1e-8:
        k1 = rhs(tau, w)
        k2 = rhs(tau + h/2, [w[i] + h/2 * k1[i] for i in range(4)])
        k3 = rhs(tau + h/2, [w[i] + h/2 * k2[i] for i in range(4)])
        k4 = rhs(tau + h,   [w[i] + h * k3[i] for i in range(4)])
        for i in range(4):
            w[i] += h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        tau += h
        times.append(tau / omega0)
        x_vals.append(w[0] * L)
        y_vals.append(w[1] * L)

    return x_vals, y_vals, times

# Расчёт для всех городов
data = []
for name, phi in cities:
    xv, yv, tv = compute_trajectory(phi)
    data.append((name, phi, xv, yv, tv))

# === Создание окна с тремя интерактивными анимациями ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Маятник Фуко без затухания (L={L} м, t_max={t_max/3600:.1f} ч)\n'
             'Начальные условия: x0=20 м, vy0=0.4 м/с', fontsize=12)

points = []
traces = []
time_texts = []

for i, (name, phi, x_vals, y_vals, times) in enumerate(data):
    ax = axes[i]
    ax.set_xlim(-x0*1.2, x0*1.2)
    ax.set_ylim(-x0*1.2, x0*1.2)
    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')
    # Информация о направлении поворота
    if phi > 0:
        direction = "против часовой"
    elif phi < 0:
        direction = "по часовой"
    else:
        direction = "отсутствует"
    ax.set_title(f'{name}\nширота {phi:.1f}°, поворот {direction}')
    ax.grid(True)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_aspect('equal')

    point, = ax.plot([], [], 'ro', markersize=4)
    trace, = ax.plot([], [], 'b-', linewidth=0.5, alpha=0.7)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
    points.append(point)
    traces.append(trace)
    time_texts.append(time_text)

# Синхронизируем длину анимации по самой короткой траектории
min_len = min(len(d[2]) for d in data)
step = max(1, min_len // 800)   # около 800 кадров для плавности
frames = range(0, min_len, step)

def update(frame):
    for i, (_, _, x_vals, y_vals, times) in enumerate(data):
        if frame < len(x_vals):
            x_show = x_vals[:frame+1]
            y_show = y_vals[:frame+1]
            points[i].set_data([x_show[-1]], [y_show[-1]])
            traces[i].set_data(x_show, y_show)
            time_texts[i].set_text(f'{times[frame]:.0f} с')
    return points + traces + time_texts

ani = animation.FuncAnimation(fig, update, frames=frames, interval=20, repeat=False)

plt.tight_layout()
plt.show()