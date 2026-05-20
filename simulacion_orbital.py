import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod



class CuerpoEspacial(ABC):
    def __init__(self, nombre, masa, posicion, velocidad):
        self.nombre = nombre
        self.masa = masa
        self.posicion = np.array(posicion, dtype=float)
        self.velocidad = np.array(velocidad, dtype=float)

    @abstractmethod
    def actualizar_estado(self, nueva_posicion, nueva_velocidad):
        pass

class Planeta(CuerpoEspacial):
    def __init__(self, nombre, masa):
        super().__init__(nombre, masa, posicion=[0.0, 0.0], velocidad=[0.0, 0.0])

    def actualizar_estado(self, nueva_posicion, nueva_velocidad):
        pass

class Nave(CuerpoEspacial):
    def __init__(self, nombre, masa, posicion, velocidad):
        super().__init__(nombre, masa, posicion, velocidad)

    def actualizar_estado(self, nueva_posicion, nueva_velocidad):
        self.posicion = np.array(nueva_posicion)
        self.velocidad = np.array(nueva_velocidad)

    def encender_retrocohetes(self, delta_v):
        magnitud_velocidad = np.linalg.norm(self.velocidad)
        if magnitud_velocidad > 0:
            direccion = self.velocidad / magnitud_velocidad
            self.velocidad -= direccion * delta_v


class SimuladorOrbital:
    def __init__(self, planeta, nave):
        self.planeta = planeta
        self.nave = nave
        self.G = 1.0  
        
        self.historial_t = []
        self.historial_pos = []
        self.historial_vel = []

    def ecuaciones_movimiento(self, t, estado):
        x, y, vx, vy = estado
        r = np.sqrt(x**2 + y**2)
        
        if r < 2.0: 
            return [0, 0, 0, 0]
            
        ax = -self.G * self.planeta.masa * x / r**3
        ay = -self.G * self.planeta.masa * y / r**3
        return [vx, vy, ax, ay]

    def simular_hasta_periastro(self, t_max_estimado, max_paso):
        
        def periastro(t, estado):
            x, y, vx, vy = estado
            return (x * vx) + (y * vy) 
        
        periastro.terminal = True 
        periastro.direction = 1 

        estado_inicial = [self.nave.posicion[0], self.nave.posicion[1], 
                          self.nave.velocidad[0], self.nave.velocidad[1]]
        
        solucion = solve_ivp(self.ecuaciones_movimiento, [0, t_max_estimado], estado_inicial, 
                             max_step=max_paso, events=periastro, method='RK45')
        
        self.historial_t.extend(solucion.t)
        self.historial_pos = solucion.y[:2, :]
        self.historial_vel = solucion.y[2:, :]
        
        self.nave.actualizar_estado(
            nueva_posicion=[solucion.y[0, -1], solucion.y[1, -1]],
            nueva_velocidad=[solucion.y[2, -1], solucion.y[3, -1]]
        )
        return solucion.t[-1]

    def simular_tramo(self, t_inicio, t_fin, max_paso):
        estado_inicial = [self.nave.posicion[0], self.nave.posicion[1], 
                          self.nave.velocidad[0], self.nave.velocidad[1]]
        
        solucion = solve_ivp(self.ecuaciones_movimiento, [t_inicio, t_fin], estado_inicial, 
                             max_step=max_paso, method='RK45')
        
        self.historial_t.extend(solucion.t[1:])
        self.historial_pos = np.hstack((self.historial_pos, solucion.y[:2, 1:]))
        self.historial_vel = np.hstack((self.historial_vel, solucion.y[2:, 1:]))
        
        self.nave.actualizar_estado(
            nueva_posicion=[solucion.y[0, -1], solucion.y[1, -1]],
            nueva_velocidad=[solucion.y[2, -1], solucion.y[3, -1]]
        )

    def calcular_energias(self):
        vel_magnitud = np.linalg.norm(self.historial_vel, axis=0)
        r_magnitud = np.linalg.norm(self.historial_pos, axis=0)
        
        energia_cinetica = 0.5 * self.nave.masa * vel_magnitud**2
        energia_potencial = - (self.G * self.planeta.masa * self.nave.masa) / r_magnitud
        energia_total = energia_cinetica + energia_potencial
        
        return self.historial_t, energia_cinetica, energia_potencial, energia_total


if __name__ == "__main__":
    tierra_gigante = Planeta("Planeta Masivo", masa=500)
    explorador = Nave("Nave Exploradora", masa=1, posicion=[-60, 30], velocidad=[7.0, 0])
    
    simulador = SimuladorOrbital(tierra_gigante, explorador)
    
   
    t_frenado = simulador.simular_hasta_periastro(t_max_estimado=30, max_paso=0.05)
    
    explorador.encender_retrocohetes(delta_v=3.0)
    
   
    simulador.simular_tramo(t_inicio=t_frenado, t_fin=250, max_paso=0.05)


    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_title("Simulación de Captura Orbital\n(Cierra esta ventana al terminar para ver las energías)")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    planeta_dibujo = plt.Circle((0, 0), 2.0, color='blue', label=tierra_gigante.nombre)
    ax.add_patch(planeta_dibujo)
    
    linea_trayectoria, = ax.plot([], [], 'g--', alpha=0.6)
    nave_dibujo, = ax.plot([], [], 'ro', markersize=6, label=explorador.nombre)
    texto_estado = ax.text(-55, 50, '', fontsize=10)
    ax.legend(loc="lower right")

    def actualizar_animacion(frame):
        f_real = min(frame * 4, len(simulador.historial_t) - 1)
        
        linea_trayectoria.set_data(simulador.historial_pos[0, :f_real], simulador.historial_pos[1, :f_real])
        nave_dibujo.set_data([simulador.historial_pos[0, f_real]], [simulador.historial_pos[1, f_real]])
        
        t_actual = simulador.historial_t[f_real]
        if t_actual < t_frenado:
            texto_estado.set_text(f"Viajando hacia el planeta... (t={t_actual:.1f})")
            texto_estado.set_color('black')
        elif t_actual < t_frenado + 2:
            texto_estado.set_text("¡MOTORES ENCENDIDOS! FRENANDO...")
            texto_estado.set_color('red')
        else:
            texto_estado.set_text("ÓRBITA ELÍPTICA ESTABLECIDA")
            texto_estado.set_color('green')
            
        return linea_trayectoria, nave_dibujo, texto_estado

    total_frames = len(simulador.historial_t) // 4
    ani = animation.FuncAnimation(fig, actualizar_animacion, 
                                  frames=total_frames, 
                                  interval=15, blit=True)
    
    plt.show() 

    print("Generando análisis de energía...")
    t, ek, ep, etotal = simulador.calcular_energias()
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, ek, label='Energía Cinética', color='blue')
    plt.plot(t, ep, label='Energía Potencial', color='orange')
    plt.plot(t, etotal, label='Energía Total', color='red', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(t_frenado, color='gray', linestyle=':', label='Punto de frenado automático')
    plt.title("Análisis de Energía: Captura Orbital Automática")
    plt.xlabel("Tiempo")
    plt.ylabel("Energía")
    plt.legend()
    plt.grid(True)
    plt.show()
