import pygame
import os
import threading
import time
from typing import Dict, Optional

class AudioManager:
    """
    Gestor de audio optimizado que maneja la carga y reproducción de música
    de fondo y efectos de sonido con caching y threading para mejor rendimiento.
    """
    
    def __init__(self):
        # Inicializar mixer de pygame si no está inicializado
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Cache de archivos de audio cargados
        self._audio_cache: Dict[str, pygame.mixer.Sound] = {}
        self._music_cache: Dict[str, str] = {}  # Solo guardamos las rutas para música
        
        # Estado actual
        self._current_music = None
        self._current_volume = 0.7
        self._music_enabled = True
        self._sfx_enabled = True
        self._sfx_volume = 0.8
        
        # Control de hilos para carga asíncrona
        self._loading_threads = []
        self._preload_lock = threading.Lock()
        
        # Rutas de audio predefinidas (archivos .mp3)
        self.audio_paths = {
            'music': {
                'home': 'game/assets/sounds/music/main.mp3',
                'level_select': 'game/assets/sounds/music/menu-levels.mp3',
                'gameplay': 'game/assets/sounds/music/gameplay_theme.mp3',
                'victory': 'game/assets/sounds/music/victory_theme.mp3',
                'defeat': 'game/assets/sounds/music/defeat_theme.mp3'
            },
            'sfx': {
                'button_hover': 'game/assets/sounds/sfx/button_hover.mp3',
                'button_click': 'game/assets/sounds/sfx/button_click.mp3',
                'transition': 'game/assets/sounds/sfx/transition.mp3',
                'goal_scored': 'game/assets/sounds/sfx/goal_scored.mp3',
                'ice_melt': 'game/assets/sounds/sfx/ice_melt.mp3',
                'environmental_heal': 'game/assets/sounds/sfx/environmental_heal.mp3'
            }
        }
        
        # Crear directorios de audio si no existen
        self._ensure_audio_directories()
        
        # Precargar audio crítico
        self._preload_critical_audio()
    
    def _ensure_audio_directories(self):
        """Crear directorios de audio si no existen"""
        try:
            music_dir = 'game/assets/sounds/music'
            sfx_dir = 'game/assets/sounds/sfx'
            
            os.makedirs(music_dir, exist_ok=True)
            os.makedirs(sfx_dir, exist_ok=True)
        except Exception as e:
            print(f"Advertencia: No se pudieron crear directorios de audio: {e}")
    
    def _preload_critical_audio(self):
        """Precargar audio crítico en un hilo separado"""
        def preload_worker():
            critical_sounds = ['button_hover', 'button_click', 'transition']
            for sound_name in critical_sounds:
                self._load_sound_effect(sound_name, async_load=False)
        
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
        self._loading_threads.append(thread)
    
    def _load_sound_effect(self, sound_name: str, async_load: bool = True) -> Optional[pygame.mixer.Sound]:
        """Cargar un efecto de sonido con cache"""
        if sound_name in self._audio_cache:
            return self._audio_cache[sound_name]
        
        if sound_name not in self.audio_paths['sfx']:
            print(f"Advertencia: Efecto de sonido '{sound_name}' no encontrado en rutas predefinidas")
            return None
        
        file_path = self.audio_paths['sfx'][sound_name]
        
        def load_worker():
            try:
                if os.path.exists(file_path):
                    with self._preload_lock:
                        if sound_name not in self._audio_cache:  # Double-check locking
                            sound = pygame.mixer.Sound(file_path)
                            sound.set_volume(self._sfx_volume)
                            self._audio_cache[sound_name] = sound
                            print(f"Audio cargado: {sound_name}")
                else:
                    print(f"Archivo de audio no encontrado: {file_path}")
            except Exception as e:
                print(f"Error cargando efecto de sonido {sound_name}: {e}")
        
        if async_load:
            thread = threading.Thread(target=load_worker, daemon=True)
            thread.start()
            self._loading_threads.append(thread)
            return None
        else:
            load_worker()
            return self._audio_cache.get(sound_name)
    
    def play_music(self, music_name: str, loop: bool = True, fade_in_ms: int = 1000):
        """
        Reproducir música de fondo con transición suave
        
        Args:
            music_name: Nombre de la música a reproducir
            loop: Si la música debe repetirse
            fade_in_ms: Duración del fade-in en milisegundos
        """
        if not self._music_enabled:
            return
        
        if music_name not in self.audio_paths['music']:
            print(f"Advertencia: Música '{music_name}' no encontrada")
            return
        
        file_path = self.audio_paths['music'][music_name]
        
        # Si es la misma música que ya está sonando, no hacer nada
        if self._current_music == music_name and pygame.mixer.music.get_busy():
            return
        
        try:
            # Detener música actual con fade-out
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.fadeout(500)
                time.sleep(0.5)  # Breve pausa para permitir fade-out
            
            # Cargar y reproducir nueva música
            if os.path.exists(file_path):
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.set_volume(self._current_volume)
                
                loops = -1 if loop else 0
                if fade_in_ms > 0:
                    pygame.mixer.music.play(loops, fade_ms=fade_in_ms)
                else:
                    pygame.mixer.music.play(loops)
                
                self._current_music = music_name
                print(f"Reproduciendo música: {music_name}")
            else:
                print(f"Archivo de música no encontrado: {file_path}")
                # Crear archivo de placeholder silencioso
                self._create_placeholder_audio(file_path)
                
        except Exception as e:
            print(f"Error reproduciendo música {music_name}: {e}")
    
    def play_sound_effect(self, sound_name: str, volume_override: Optional[float] = None):
        """
        Reproducir efecto de sonido
        
        Args:
            sound_name: Nombre del efecto a reproducir
            volume_override: Volumen específico para este sonido (0.0-1.0)
        """
        if not self._sfx_enabled:
            return
        
        # Cargar sonido si no está en cache
        if sound_name not in self._audio_cache:
            sound = self._load_sound_effect(sound_name, async_load=False)
            if not sound:
                return
        
        try:
            sound = self._audio_cache[sound_name]
            
            # Aplicar volumen específico si se proporciona
            if volume_override is not None:
                sound.set_volume(volume_override)
            else:
                sound.set_volume(self._sfx_volume)
            
            sound.play()
            
        except Exception as e:
            print(f"Error reproduciendo efecto {sound_name}: {e}")
    
    def stop_music(self, fade_out_ms: int = 1000):
        """Detener música con fade-out"""
        try:
            if pygame.mixer.music.get_busy():
                if fade_out_ms > 0:
                    pygame.mixer.music.fadeout(fade_out_ms)
                else:
                    pygame.mixer.music.stop()
                self._current_music = None
        except Exception as e:
            print(f"Error deteniendo música: {e}")
    
    def set_music_volume(self, volume: float):
        """Establecer volumen de música (0.0-1.0)"""
        self._current_volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(self._current_volume)
        except Exception as e:
            print(f"Error estableciendo volumen de música: {e}")
    
    def set_sfx_volume(self, volume: float):
        """Establecer volumen de efectos de sonido (0.0-1.0)"""
        self._sfx_volume = max(0.0, min(1.0, volume))
        
        # Actualizar volumen de sonidos cargados
        with self._preload_lock:
            for sound in self._audio_cache.values():
                try:
                    sound.set_volume(self._sfx_volume)
                except Exception as e:
                    print(f"Error actualizando volumen de efecto: {e}")
    
    def toggle_music(self):
        """Alternar música encendida/apagada"""
        if self._music_enabled:
            # Si la música está habilitada, la desactivamos
            self._music_enabled = False
            pygame.mixer.music.pause()  # Pausar la música en lugar de detenerla
        else:
            # Si la música está deshabilitada, la activamos
            self._music_enabled = True
            
            if pygame.mixer.music.get_busy():
                # Si la música estaba en pausa, la reanudamos
                pygame.mixer.music.unpause()
            else:
                # Si la música no estaba cargada o se detuvo completamente, la reiniciamos
                if self._current_music:
                    self.play_music(self._current_music)

    def toggle_sfx(self):
        """Alternar efectos de sonido encendidos/apagados"""
        self._sfx_enabled = not self._sfx_enabled
        return self._sfx_enabled
    
    def preload_audio_for_screen(self, screen_name: str):
        """Precargar audio específico para una pantalla"""
        def preload_worker():
            # Mapeo de pantallas a audio que necesitan
            screen_audio_map = {
                'home': ['button_hover', 'button_click', 'transition'],
                'level_select': ['button_hover', 'button_click', 'transition'],
                'gameplay': ['goal_scored', 'ice_melt', 'environmental_heal']
            }
            
            if screen_name in screen_audio_map:
                for sound_name in screen_audio_map[screen_name]:
                    self._load_sound_effect(sound_name, async_load=False)
        
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
        self._loading_threads.append(thread)
    
    def _create_placeholder_audio(self, file_path: str):
        """Crear archivo de audio silencioso como placeholder"""
        try:
            # Crear 1 segundo de silencio usando pygame
            sample_rate = 22050
            duration = 1.0
            samples = int(sample_rate * duration)
            
            # Crear array de silencio estéreo
            import array
            silence_data = array.array('h', [0] * (samples * 2))  # Estéreo, 16-bit
            
            # Crear superficie de sonido desde los datos de silencio
            silence_sound = pygame.sndarray.make_sound(silence_data.tolist())
            
            # Crear directorio si no existe
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            
            # Crear archivo WAV simple como placeholder
            placeholder_path = file_path.replace('.mp3', '_placeholder.wav')
            
            # Usar una implementación simple para crear archivo WAV
            try:
                # Escribir encabezado WAV básico
                with open(placeholder_path, 'wb') as f:
                    # Encabezado WAV (44 bytes)
                    f.write(b'RIFF')  # ChunkID
                    f.write((36 + samples * 2).to_bytes(4, 'little'))  # ChunkSize
                    f.write(b'WAVE')  # Format
                    f.write(b'fmt ')  # Subchunk1ID
                    f.write((16).to_bytes(4, 'little'))  # Subchunk1Size
                    f.write((1).to_bytes(2, 'little'))  # AudioFormat (PCM)
                    f.write((1).to_bytes(2, 'little'))  # NumChannels (mono)
                    f.write(sample_rate.to_bytes(4, 'little'))  # SampleRate
                    f.write((sample_rate * 2).to_bytes(4, 'little'))  # ByteRate
                    f.write((2).to_bytes(2, 'little'))  # BlockAlign
                    f.write((16).to_bytes(2, 'little'))  # BitsPerSample
                    f.write(b'data')  # Subchunk2ID
                    f.write((samples * 2).to_bytes(4, 'little'))  # Subchunk2Size
                    
                    # Datos de audio (silencio)
                    for _ in range(samples):
                        f.write((0).to_bytes(2, 'little', signed=True))
                
                print(f"Creado archivo de audio placeholder: {placeholder_path}")
                
                # Actualizar la ruta en el cache para usar el placeholder
                for category, sounds in self.audio_paths.items():
                    for sound_name, sound_path in sounds.items():
                        if sound_path == file_path:
                            self.audio_paths[category][sound_name] = placeholder_path
                            break
                
            except Exception as e:
                print(f"Error creando archivo WAV placeholder: {e}")
                # Como último recurso, crear un archivo vacío
                try:
                    with open(placeholder_path, 'w') as f:
                        f.write('')
                    print(f"Creado archivo placeholder vacío: {placeholder_path}")
                except Exception as e2:
                    print(f"Error creando archivo placeholder vacío: {e2}")
                
        except Exception as e:
            print(f"Error creando placeholder de audio: {e}")
    
    def cleanup(self):
        """Limpiar recursos de audio"""
        try:
            # Detener toda la música
            pygame.mixer.music.stop()
            
            # Detener todos los efectos
            pygame.mixer.stop()
            
            # Esperar a que terminen los hilos de carga
            for thread in self._loading_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            # Limpiar cache
            self._audio_cache.clear()
            self._music_cache.clear()
            
            print("AudioManager limpiado")
            
        except Exception as e:
            print(f"Error durante limpieza de AudioManager: {e}")
    
    def get_status(self) -> dict:
        """Obtener estado actual del audio"""
        return {
            'music_enabled': self._music_enabled,
            'sfx_enabled': self._sfx_enabled,
            'music_volume': self._current_volume,
            'sfx_volume': self._sfx_volume,
            'current_music': self._current_music,
            'is_music_playing': pygame.mixer.music.get_busy(),
            'cached_sounds': len(self._audio_cache),
            'loading_threads': len([t for t in self._loading_threads if t.is_alive()])
        }

    def get_music_volume(self):
        """Obtener el volumen actual de la música"""
        return self._current_volume

    def get_sfx_volume(self):
        """Obtener el volumen actual de los efectos"""
        return self._sfx_volume

    def is_music_enabled(self):
        """Verificar si la música está habilitada"""
        return self._music_enabled

    def is_sfx_enabled(self):
        """Verificar si los efectos están habilitados"""
        return self._sfx_enabled

    def save_audio_settings_to_profile(self, save_system):
        """Guardar configuración de audio en el perfil actual del usuario"""
        if not save_system or not save_system.current_profile:
            print("No hay perfil activo para guardar configuración de audio")
            return False
            
        settings = {
            "music_volume": self._current_volume,
            "sfx_volume": self._sfx_volume,
            "music_enabled": self._music_enabled,
            "sfx_enabled": self._sfx_enabled
        }
        
        # Actualizar la configuración en el perfil
        return save_system.update_settings(settings)

    def load_audio_settings_from_profile(self, save_system):
        """Cargar configuración de audio desde el perfil actual del usuario"""
        if not save_system or not save_system.current_profile:
            print("No hay perfil activo para cargar configuración de audio")
            return False
        
        # Obtener configuración del perfil
        profile = save_system.get_current_profile_data()
        if profile and "settings" in profile:
            # Aplicar configuración de audio
            self._current_volume = profile["settings"].get("music_volume", 0.7)
            self._sfx_volume = profile["settings"].get("sfx_volume", 0.8)
            self._music_enabled = profile["settings"].get("music_enabled", True)
            self._sfx_enabled = profile["settings"].get("sfx_enabled", True)
            
            # Aplicar volumen actual a la música si está sonando
            pygame.mixer.music.set_volume(self._current_volume)
            
            # Actualizar volumen de efectos cargados
            with self._preload_lock:
                for sound in self._audio_cache.values():
                    try:
                        sound.set_volume(self._sfx_volume)
                    except Exception as e:
                        print(f"Error actualizando volumen de efecto: {e}")
            
            # Si la música está desactivada, detenerla
            if not self._music_enabled and pygame.mixer.music.get_busy():
                self.stop_music()
            # Si la música está activada y hay una pista actual, asegurar que suena
            elif self._music_enabled and not pygame.mixer.music.get_busy() and self._current_music:
                self.play_music(self._current_music)
                
            return True
        return False

# Instancia global del gestor de audio
audio_manager = AudioManager()
