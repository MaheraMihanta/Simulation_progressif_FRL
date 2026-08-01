from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

# 1. Connexion correcte à l'API
client = RemoteAPIClient()
sim = client.require('sim')

# 2. Chemins absolus indexés basés sur votre arbre CoppeliaSim
# [0] correspond au premier élément trouvé, [1] au deuxième, etc.
"""joint_names = [
    '/NiryoOne/Joint',
    '/NiryoOne/Joint/Link/Joint',
    '/NiryoOne/Joint/Link/Joint[1]',
    '/NiryoOne/Joint/Link/Joint[2]',
    '/NiryoOne/Joint/Link/Joint[3]',
    '/NiryoOne/Joint/Link/Joint[4]'
]
"""
joint_names = [
    '/NiryoOne/Joint',
    '/NiryoOne/Joint/Link/Joint',
    '/NiryoOne/Joint/Link/Joint/Link/Joint',
    '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint',
    '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint',
    '/NiryoOne/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint/Link/Joint'
]

# 3. Récupération des handles
joint_handles = []
try:
    for name in joint_names:
        handle = sim.getObject(name)
        joint_handles.append(handle)
        print(f"Handle récupéré pour {name} : {handle}")
        
    print("\n👉 Tous les handles ont été récupérés avec succès !")

except Exception as e:
    print(f"\n❌ Erreur. Détails : {e}")


try:
    # 1. Démarrer la simulation de CoppeliaSim depuis Python
    print("\nDémarrage de la simulation...")
    sim.startSimulation()
    time.sleep(0.5)  # Petit temps de pause pour laisser la physique s'initialiser
    
    # 2. Définir une cible de position pour l'articulation 1 (Base)
    # CoppeliaSim utilise les RADIANS (45 degrés = ~0.785 rad)
    target_angle = 0.785 
    print(f"Envoi de la commande : incliner l'axe 1 à {target_angle} rad (45°)...")
    
    # joint_handles[0] correspond au premier handle récupéré (53)
    sim.setJointTargetPosition(joint_handles[0], target_angle)
    
    # 3. Laisser le robot bouger pendant 3 secondes
    print("Mouvement en cours...")
    time.sleep(3)
    
    # 4. Lire la position réelle pour confirmer le mouvement
    current_angle = sim.getJointPosition(joint_handles[0])
    print(f"Position réelle finale de l'axe 1 : {current_angle:.3f} rad")
    
    # 5. Arrêter proprement la simulation
    print("Arrêt de la simulation.")
    sim.stopSimulation()

except Exception as e:
    print(f"Une erreur est survenue pendant le mouvement : {e}")
