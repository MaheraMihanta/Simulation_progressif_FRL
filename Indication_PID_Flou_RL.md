J' ai remarqué que pour des degrés de libertés faible (2 ou 3) la complexité de calcul en utilisant la logique floue montre déjà des demandes en ressources considérables, il faut presque 700 règles floues. Et donc pour arriver aux 6DDL, il faudra plus de 500.000 règles, ce qui est inenvisagable.

## Comment donc réorienter notre étude? en voulant mettre en oeuvre un controleur flou-RL, et le RL donne un résidu de commnande.

1. Il est clair que la logique floue n'est pas une option viable pour les futurs extensions de DDL que l'on prévoit, j'ai fait quelques recherches et une piste à explorer : Combler l'inaptitude des contrôleurs PID par une adaptation dynamique des Kp, Ki et Kd, et on obtient un compromis traitable, et c'est là qu'on ajoutera les commandes résiduelles par RL.

2. Je veux que vous évaluer cette option, et pour la suite, commencez l'implémentation par l'ajout d'un 4ème DDL au Robot 3DDL actuel déjà implémenté ici dans ce dossier.

3. Simulez d'abord PID, PID avec réglage dynamique floue, et c'est là que vous introduirez ensuite le RL.