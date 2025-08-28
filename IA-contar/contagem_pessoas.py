import cv2
import torch
import numpy as np
from collections import defaultdict
import math

class ObjectTracker:
    def __init__(self, frame_width, frame_height, max_distance=80, max_frames_lost=15, entrance_zone_size=40):
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.trackers = {}
        self.next_id = 0
        self.counters = {'person': 0, 'car': 0, 'motorcycle': 0}
        
        # Definir zonas de entrada (bordas da tela) - agora é parâmetro ajustável
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.entrance_zone_size = entrance_zone_size  # pixels da borda (ajustável)
        
        # Rastrear objetos que já foram contados
        self.counted_objects = set()
        
        # Mostrar configuração
        print(f"Tracker configurado:")
        print(f"- Distância máxima: {max_distance} pixels")
        print(f"- Frames perdidos: {max_frames_lost}")
        print(f"- Zona de entrada: {entrance_zone_size} pixels")
        
    def is_in_entrance_zone(self, x, y):
        """Verifica se um objeto está na zona de entrada (próximo às bordas)"""
        return (x < self.entrance_zone_size or 
                x > self.frame_width - self.entrance_zone_size or
                y < self.entrance_zone_size or 
                y > self.frame_height - self.entrance_zone_size)
    
    def is_in_center_zone(self, x, y):
        """Verifica se um objeto está na zona central (longe das bordas)"""
        margin = self.entrance_zone_size * 2
        return (margin < x < self.frame_width - margin and 
                margin < y < self.frame_height - margin)
        
    def calculate_distance(self, pos1, pos2):
        """Calcula a distância euclidiana entre dois pontos"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update(self, detections_by_class):
        """Atualiza os trackers com novas detecções"""
        current_frame_trackers = {}
        
        # Para cada classe de objeto
        for class_name, detections in detections_by_class.items():
            current_positions = []
            
            # Extrair posições centrais das detecções
            for _, detection in detections.iterrows():
                center_x = (detection['xmin'] + detection['xmax']) / 2
                center_y = (detection['ymin'] + detection['ymax']) / 2
                current_positions.append((center_x, center_y, detection))
            
            # Encontrar correspondências com trackers existentes
            used_detections = set()
            
            for tracker_id, tracker_info in self.trackers.items():
                if tracker_info['class'] != class_name:
                    continue
                    
                best_match = None
                best_distance = float('inf')
                best_idx = -1
                
                # Encontrar a detecção mais próxima
                for idx, (x, y, detection) in enumerate(current_positions):
                    if idx in used_detections:
                        continue
                        
                    distance = self.calculate_distance(tracker_info['position'], (x, y))
                    
                    if distance < self.max_distance and distance < best_distance:
                        best_distance = distance
                        best_match = (x, y, detection)
                        best_idx = idx
                
                if best_match:
                    # Atualizar tracker existente
                    new_x, new_y = best_match[0], best_match[1]
                    
                    # Verificar se deve contar o objeto
                    should_count = False
                    if tracker_id not in self.counted_objects:
                        # Se começou na zona de entrada e agora está na zona central
                        if (tracker_info.get('started_at_entrance', False) and 
                            self.is_in_center_zone(new_x, new_y)):
                            should_count = True
                        # Ou se estava na zona de entrada e se moveu significativamente
                        elif (self.is_in_entrance_zone(tracker_info['position'][0], tracker_info['position'][1]) and
                              not self.is_in_entrance_zone(new_x, new_y) and
                              self.calculate_distance(tracker_info['initial_position'], (new_x, new_y)) > 80):
                            should_count = True
                    
                    if should_count:
                        self.counters[class_name] += 1
                        self.counted_objects.add(tracker_id)
                        print(f"Novo {class_name} contado (ID: {tracker_id})! Total: {self.counters[class_name]}")
                    
                    current_frame_trackers[tracker_id] = {
                        'position': (new_x, new_y),
                        'class': class_name,
                        'frames_lost': 0,
                        'bbox': (int(best_match[2]['xmin']), int(best_match[2]['ymin']), 
                                int(best_match[2]['xmax']), int(best_match[2]['ymax'])),
                        'initial_position': tracker_info.get('initial_position', (new_x, new_y)),
                        'started_at_entrance': tracker_info.get('started_at_entrance', False)
                    }
                    used_detections.add(best_idx)
                else:
                    # Manter tracker mas aumentar frames_lost
                    if tracker_info['frames_lost'] < self.max_frames_lost:
                        current_frame_trackers[tracker_id] = {
                            'position': tracker_info['position'],
                            'class': tracker_info['class'],
                            'frames_lost': tracker_info['frames_lost'] + 1,
                            'bbox': tracker_info.get('bbox', (0, 0, 0, 0)),
                            'initial_position': tracker_info.get('initial_position', tracker_info['position']),
                            'started_at_entrance': tracker_info.get('started_at_entrance', False)
                        }
            
            # Criar novos trackers para detecções não utilizadas
            for idx, (x, y, detection) in enumerate(current_positions):
                if idx not in used_detections:
                    started_at_entrance = self.is_in_entrance_zone(x, y)
                    
                    # Só conta imediatamente se o objeto aparece no centro da tela
                    # (provavelmente já estava lá antes da detecção começar)
                    should_count_immediately = False
                    if not started_at_entrance and self.is_in_center_zone(x, y):
                        # Verifica se não há outros objetos muito próximos que já foram contados
                        too_close_to_counted = False
                        for existing_id, existing_tracker in current_frame_trackers.items():
                            if (existing_tracker['class'] == class_name and 
                                existing_id in self.counted_objects and
                                self.calculate_distance((x, y), existing_tracker['position']) < self.max_distance * 1.5):
                                too_close_to_counted = True
                                break
                        
                        if not too_close_to_counted:
                            should_count_immediately = True
                    
                    if should_count_immediately:
                        self.counters[class_name] += 1
                        self.counted_objects.add(self.next_id)
                        print(f"Novo {class_name} detectado no centro (ID: {self.next_id})! Total: {self.counters[class_name]}")
                    
                    current_frame_trackers[self.next_id] = {
                        'position': (x, y),
                        'class': class_name,
                        'frames_lost': 0,
                        'bbox': (int(detection['xmin']), int(detection['ymin']), 
                                int(detection['xmax']), int(detection['ymax'])),
                        'initial_position': (x, y),
                        'started_at_entrance': started_at_entrance
                    }
                    self.next_id += 1
        
        self.trackers = current_frame_trackers
        return self.trackers
    
    def get_counters(self):
        """Retorna os contadores atuais"""
        return self.counters.copy()
    
    def get_current_count(self):
        """Retorna a contagem atual de objetos na tela"""
        current_count = {'person': 0, 'car': 0, 'motorcycle': 0}
        for tracker_info in self.trackers.values():
            if tracker_info['frames_lost'] == 0:
                current_count[tracker_info['class']] += 1
        return current_count
    
    def draw_zones(self, frame):
        """Desenha as zonas de entrada na tela para visualização"""
        height, width = frame.shape[:2]
        
        # Zona de entrada (bordas) - linha vermelha
        cv2.rectangle(frame, (0, 0), (self.entrance_zone_size, height), (0, 0, 255), 2)  # Esquerda
        cv2.rectangle(frame, (width - self.entrance_zone_size, 0), (width, height), (0, 0, 255), 2)  # Direita
        cv2.rectangle(frame, (0, 0), (width, self.entrance_zone_size), (0, 0, 255), 2)  # Topo
        cv2.rectangle(frame, (0, height - self.entrance_zone_size), (width, height), (0, 0, 255), 2)  # Base
        
        # Zona central - linha verde
        margin = self.entrance_zone_size * 2
        cv2.rectangle(frame, (margin, margin), (width - margin, height - margin), (0, 255, 0), 1)

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# URL RTSP com autenticação escapada
source = "rtsp://admin:%40Well32213115@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0"

# Para testar com webcam, descomente a linha abaixo:
# source = 0

# Iniciar captura de vídeo
cap = cv2.VideoCapture(source)

# Verifique se a captura foi iniciada com sucesso
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Obter dimensões do frame
ret, test_frame = cap.read()
if ret:
    frame_height, frame_width = test_frame.shape[:2]
    # ESCOLHA UMA DAS CONFIGURAÇÕES ABAIXO:
    
    # CONFIGURAÇÃO 1 - ESTACIONAMENTO (objetos grandes, câmera próxima)
    # tracker = ObjectTracker(frame_width, frame_height, max_distance=120, max_frames_lost=30, entrance_zone_size=60)
    
    # CONFIGURAÇÃO 2 - RUA MOVIMENTADA (objetos médios, movimento rápido)
    tracker = ObjectTracker(frame_width, frame_height, max_distance=80, max_frames_lost=15, entrance_zone_size=40)
    
    # CONFIGURAÇÃO 3 - CORREDOR/PORTA (objetos pequenos, câmera distante)
    # tracker = ObjectTracker(frame_width, frame_height, max_distance=50, max_frames_lost=25, entrance_zone_size=30)
else:
    print("Erro ao ler o primeiro frame")
    exit()

# Mapeamento de classes
class_names = {0: 'person', 2: 'car', 3: 'motorcycle'}
class_colors = {'person': (0, 255, 0), 'car': (0, 0, 255), 'motorcycle': (255, 0, 0)}

show_zones = False  # Variável para mostrar/esconder zonas

print("Sistema de contagem iniciado.")
print("CONFIGURAÇÕES DISPONÍVEIS:")
print("1. ESTACIONAMENTO: max_distance=120, max_frames_lost=30, entrance_zone_size=60")
print("2. RUA MOVIMENTADA: max_distance=80, max_frames_lost=15, entrance_zone_size=40") 
print("3. CORREDOR/PORTA: max_distance=50, max_frames_lost=25, entrance_zone_size=30")
print("\nControlos:")
print("- 'q': Sair")
print("- 'r': Resetar contadores") 
print("- 'z': Mostrar/esconder zonas de detecção")
print("- '1', '2', '3': Trocar configuração em tempo real")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # Organizar detecções por classe
    detections_by_class = {}
    for class_id, class_name in class_names.items():
        class_detections = detections[detections['class'] == class_id]
        detections_by_class[class_name] = class_detections

    # Atualizar tracker
    current_trackers = tracker.update(detections_by_class)

    # Desenhar zonas se habilitado
    if show_zones:
        tracker.draw_zones(frame)

    # Obter contadores
    total_counters = tracker.get_counters()
    current_counters = tracker.get_current_count()

    # Desenhar informações na tela
    cv2.putText(frame, f'TOTAL CONTADO:', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Pessoas: {total_counters["person"]}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Carros: {total_counters["car"]}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Motos: {total_counters["motorcycle"]}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f'NA TELA AGORA:', (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Pessoas: {current_counters["person"]}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Carros: {current_counters["car"]}', (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Motos: {current_counters["motorcycle"]}', (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Desenhar bounding boxes
    for tracker_id, tracker_info in current_trackers.items():
        if tracker_info['frames_lost'] == 0:
            class_name = tracker_info['class']
            color = class_colors[class_name]
            bbox = tracker_info['bbox']
            
            # Cor diferente para objetos já contados
            if tracker_id in tracker.counted_objects:
                # Mais escuro para objetos já contados
                color = tuple(int(c * 0.7) for c in color)
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Status do objeto
            status = "✓" if tracker_id in tracker.counted_objects else "?"
            cv2.putText(frame, f'{class_name} #{tracker_id} {status}', 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Instruções
    cv2.putText(frame, "q: sair | r: reset | z: zonas | 1,2,3: configs", 
               (frame.shape[1] - 400, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Contador Inteligente - Pessoas, Carros e Motos", frame)

    # Controles
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracker = ObjectTracker(frame_width, frame_height, max_distance=80, max_frames_lost=15, entrance_zone_size=40)
        print("Contadores resetados!")
    elif key == ord('z'):
        show_zones = not show_zones
        print(f"Zonas de detecção: {'LIGADAS' if show_zones else 'DESLIGADAS'}")
    elif key == ord('1'):
        # Configuração Estacionamento
        tracker = ObjectTracker(frame_width, frame_height, max_distance=120, max_frames_lost=30, entrance_zone_size=60)
        print("Configuração alterada para: ESTACIONAMENTO")
    elif key == ord('2'):
        # Configuração Rua Movimentada
        tracker = ObjectTracker(frame_width, frame_height, max_distance=80, max_frames_lost=15, entrance_zone_size=40)
        print("Configuração alterada para: RUA MOVIMENTADA")
    elif key == ord('3'):
        # Configuração Corredor/Porta
        tracker = ObjectTracker(frame_width, frame_height, max_distance=50, max_frames_lost=25, entrance_zone_size=30)
        print("Configuração alterada para: CORREDOR/PORTA")

# Exibir contagem final
final_counters = tracker.get_counters()
print("\n=== CONTAGEM FINAL ===")
print(f"Total de pessoas: {final_counters['person']}")
print(f"Total de carros: {final_counters['car']}")
print(f"Total de motos: {final_counters['motorcycle']}")

cap.release()
cv2.destroyAllWindows()


# 2. RUA MOVIMENTADA 🛣️
# pythonmax_distance=80, max_frames_lost=15, entrance_zone_size=40

# Cenário: Câmera em poste monitorando rua/avenida
# Características: Carros médios, movimento constante
# Por que: Movimento rápido, objetos saem logo da tela