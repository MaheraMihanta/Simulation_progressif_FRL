## Objectif

démontrer par simulation (avec python d'abord, et coppeliaSim ensuite) le principe et l'applicabilité de l'apprentissage par renforcement dans le domaine des bras robotique, l'objectif ultime est un bras 6DDL, mais je veux qu'on parte de 2DDL d'abord, et qu'on montre que les principes sont fontionnels, à savoir : les états (valeurs d'états / state value; action; reward, return, discount return, policy); j'ai déjà assimilé les bases mathématiques, je 'entre maintenant dans les algorithmes (recherche de l'optimalité de l'équation de Bellman), donc on teste les principes et les théories en 2DDL, puis en 3DDL et ainsi de suite.

## Point d'innovation et par rapport aux technologies et techinques déjà exsitants

1. Pour commander le bras (Atteinte d'un point particulier/précis; ou suivi d'une trajectoire), on va commencer par implémenter les techniques anciennes : régulateur PID, commande robuste, bref les techniques d'asservissement.

2. on reproduit les même scénarios, mais avec un contrôleur flou (utilisant la logique floue)

3. Faire une comparaison éventuelle entre ces deux techniques

4. Introduire l'apprentissage par renforcement, combiné avec la logique floue, c'est ici que vous allez faire une expertise approfondie, comment combiner ces deux techniques?

## Remarques
1. Partez toujours avec les outils simple et facile d'accès comme python et les bibliothèques qui sont déjà installé, je crois que Pyro est aussi présent si je ne me trompe.

2. Il faut donc formuler mathématiquement les équations régissant les mouvements et les comportements du robot (effecteur, moteur pas à pas pour les articulations, les masses, ...)