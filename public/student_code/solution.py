import networkx as nx
from typing import Dict, List, Literal

from public.lib.interfaces import CityGraph, ProxyData, PolicyResult
from public.student_code.convert_to_df import convert_edge_data_to_df, convert_node_data_to_df

class EvacuationPolicy:
    """
    Tu implementación de la política de evacuación.
    Esta es la clase que necesitas implementar para resolver el problema de evacuación.
    """
    
    def __init__(self):
        """Inicializa tu política de evacuación"""
        self.policy_type = "policy_1"  # Política por defecto
        
    def set_policy(self, policy_type: Literal["policy_1", "policy_2", "policy_3", "policy_4"]):
        """
        Selecciona la política a utilizar
        Args:
            policy_type: Tipo de política a utilizar
                - "policy_1": Política básica sin uso de proxies
                - "policy_2": Política usando proxies y sus descripciones
                - "policy_3": Política usando datos de simulaciones previas
                - "policy_4": Política personalizada
        """
        self.policy_type = policy_type
    
    def plan_evacuation(self, city: CityGraph, proxy_data: ProxyData, 
                       max_resources: int) -> PolicyResult:
        """
        Planifica la ruta de evacuación y la asignación de recursos.
        
        Args:
            city: El layout de la ciudad
                 - city.graph: Grafo NetworkX con el layout de la ciudad
                 - city.starting_node: Tu posición inicial
                 - city.extraction_nodes: Lista de puntos de extracción posibles
                 
            proxy_data: Información sobre el ambiente
                 - proxy_data.node_data[node_id]: Dict con indicadores de nodos
                 - proxy_data.edge_data[(node1,node2)]: Dict con indicadores de aristas
                 
            max_resources: Máximo total de recursos que puedes asignar
            
        Returns:
            PolicyResult con:
            - path: List[int] - Lista de IDs de nodos formando tu ruta de evacuación
            - resources: Dict[str, int] - Cuántos recursos de cada tipo llevar:
                       {'explosives': x, 'ammo': y, 'radiation_suits': z}
                       donde x + y + z <= max_resources
        """
        # print(f'City graph: {city.graph} \n')
        # print(f'City starting_node: {city.starting_node}\n')
        # print(f'City extraction_nodes: {city.extraction_nodes}\n')
        # print(f'Proxy node_data: {proxy_data.node_data} \n \n')
        # print(f'Proxy edge_data: {proxy_data.edge_data} \n \n')
        # print(f'Max Resources: {max_resources} \n \n')
        
        
        self.policy_type = "policy_1" # TODO: Cambiar a "policy_2" para probar la política 2, y asi sucesivamente
        
        if self.policy_type == "policy_1":
            return self._policy_1(city, max_resources)
        elif self.policy_type == "policy_2":
            return self._policy_2(city, proxy_data, max_resources)
        elif self.policy_type == "policy_3":
            return self._policy_3(city, proxy_data, max_resources)
        else:  # policy_4
            return self._policy_4(city, proxy_data, max_resources)
    
    def _policy_1(self, city: CityGraph, max_resources: int) -> PolicyResult:
        """
        Política 1: Estrategia básica sin uso de proxies.
        Solo utiliza información básica de nodos y aristas para tomar decisiones.
        
        Esta política debe:
        - NO utilizar los proxies
        - Solo usar información básica del grafo (nodos, aristas, pesos)
        - Implementar una estrategia válida para cualquier ciudad
        """
        # TODO: Implementa tu solución aquí
        target = city.extraction_nodes[0]
        
        try:
            path = nx.shortest_path(city.graph, city.starting_node, target, 
                                  weight='weight')
        except nx.NetworkXNoPath:
            path = [city.starting_node]
            
        resources = {
            'explosives': max_resources // 3,
            'ammo': max_resources // 3,
            'radiation_suits': max_resources // 3
        }
        
        return PolicyResult(path, resources)
    
    def _policy_2(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política que minimiza el hazard_gradient a lo largo de la ruta.
        Primero verifica si es posible llegar a algún punto de extracción.
        
        Esta política:
        - Verifica la conectividad del grafo
        - Se enfoca en minimizar el hazard_gradient
        - Distribuye recursos según la prioridad establecida
        """
        # Verificamos si es posible llegar a alguno de los puntos de extracción
        reachable_targets = []
        for target in city.extraction_nodes:
            try:
                # Intentamos encontrar cualquier camino (sin considerar pesos)
                path = nx.shortest_path(city.graph, city.starting_node, target)
                reachable_targets.append(target)
            except nx.NetworkXNoPath:
                continue
        
        # Si no podemos llegar a ningún punto de extracción, imprimimos mensaje y devolvemos camino vacío
        if not reachable_targets:
            return PolicyResult([city.starting_node], {
                'radiation_suits':0,
                'ammo':0,
                'explosives':0
            })
        
        # Creamos un grafo ponderado donde el peso es el hazard_gradient
        hazard_graph = city.graph.copy()
        
        # Asignamos pesos basados únicamente en el hazard_gradient
        for u, v, data in hazard_graph.edges(data=True):
            edge_key = f"{u}_{v}"
            base_weight = data.get('weight', 1)
            
            if edge_key in proxy_data.edge_data:
                # Usamos el hazard_gradient como peso principal
                hazard_value = proxy_data.edge_data[edge_key].get("hazard_gradient", 0)
                
                # Añadimos un pequeño valor constante para evitar pesos de cero
                new_weight = base_weight * (1 + hazard_value * 10 + 0.01)
                hazard_graph[u][v]['weight'] = new_weight
        
        # Encontramos el camino más corto (con menor hazard_gradient acumulado)
        best_path = None
        best_path_cost = float('inf')
        
        for target in reachable_targets:
            path = nx.shortest_path(hazard_graph, city.starting_node, target, weight='weight')
            path_cost = nx.path_weight(hazard_graph, path, weight='weight')
            
            if path_cost < best_path_cost:
                best_path = path
                best_path_cost = path_cost
        
        # Distribuimos recursos según la prioridad establecida
        # Trajes de radiación (45%), munición (35%), explosivos (20%)
        resources = {
            'radiation_suits': int(max_resources * 0.45),
            'ammo': int(max_resources * 0.393),
            'explosives': max_resources - int(max_resources * 0.45) - int(max_resources * 0.393)
        }
        
        
        # Ajustamos para asegurar que sumamos exactamente max_resources
        adjustment = max_resources - sum(resources.values())
        
        # Si hay ajuste necesario, lo asignamos prioritariamente a trajes de radiación
        if adjustment > 0:
            resources['radiation_suits'] += adjustment
        
        return PolicyResult(best_path, resources)
    def _policy_3(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 3: Estrategia usando datos de simulaciones previas.
        Utiliza estadísticas básicas de simulaciones anteriores para mejorar la toma de decisiones.
        
        Esta política debe:
        - Utilizar datos de simulaciones previas
        - Implementar mejoras basadas en estadísticas básicas
        - NO usar modelos de machine learning
        """
        # TODO: Implementa tu solución aquí
        # Aquí deberías cargar y analizar datos de simulaciones previas
        
        target = city.extraction_nodes[0]
        
        try:
            path = nx.shortest_path(city.graph, city.starting_node, target, 
                                  weight='weight')
        except nx.NetworkXNoPath:
            path = [city.starting_node]
            
        resources = {
            'explosives': max_resources // 3,
            'ammo': max_resources // 3,
            'radiation_suits': max_resources // 3
        }
        
        return PolicyResult(path, resources)
    
    def _policy_4(self, city: CityGraph, proxy_data: ProxyData, max_resources: int) -> PolicyResult:
        """
        Política 4: Estrategia personalizada.
        Implementa tu mejor estrategia usando cualquier recurso disponible.
        
        Esta política puede:
        - Usar cualquier técnica o recurso que consideres apropiado
        - Implementar estrategias avanzadas de tu elección
        """
        # TODO: Implementa tu solución aquí
        proxy_data_nodes_df = convert_node_data_to_df(proxy_data.node_data)
        proxy_data_edges_df = convert_edge_data_to_df(proxy_data.edge_data)
        
        #print(f'\n Node Data: \n {proxy_data_nodes_df}')
        #print(f'\n Edge Data: \n {proxy_data_edges_df}')
        
        target = city.extraction_nodes[0]
        
        try:
            path = nx.shortest_path(city.graph, city.starting_node, target, 
                                  weight='weight')
        except nx.NetworkXNoPath:
            path = [city.starting_node]
            
        resources = {
            'explosives': max_resources // 3,
            'ammo': max_resources // 3,
            'radiation_suits': max_resources // 3
        }
        
        return PolicyResult(path, resources)
    
    
    