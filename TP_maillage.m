%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                  IMPORTATION DES IMAGES                 %
%                                                         %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear;
close all;
% Nombre d'images utilisees
nb_images = 36; 

% chargement des images
for i = 1:nb_images
    if i<=10
        nom = sprintf('images/viff.00%d.ppm',i-1);
    else
        nom = sprintf('images/viff.0%d.ppm',i-1);
    end
    % im est une matrice de dimension 4 qui contient 
    % l'ensemble des images couleur de taille : nb_lignes x nb_colonnes x nb_canaux 
    % im est donc de dimension nb_lignes x nb_colonnes x nb_canaux x nb_images
    im(:,:,:,i) = imread(nom); 
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                 CALCULS DES SUPERPIXELS                 %
%        Appel à la fonction kmeans_segmentation          %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% On définit nos valeurs clés  N, K et S
[n,m,~] = size(im(:,:,:,16));
N = n*m;
K = 450;
S = sqrt(N/K); % Espacement entre les centres
n_part = floor(n/S); % nombre de lignes de centres
m_part = floor(m/S); % nombre de colonnes de centres

% On inistialise notre matrices des centres avec K centres
Centre = ones(nb_images, K, 5);

% On déclare nos matrices labels et distance
labels = zeros(nb_images,n,m);
distances = inf(n,m);

% Paramètres
facteur=20;
total_iter=5;

% Appel à la fonction kmeans_segmentation pour retourner les matrice de centres, les labels après les 5 itérations
[labels, distances, Centre, K_reel] = kmeans_segmentation(im, n ,m , K, nb_images,  Centre, S, facteur, total_iter);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                    TEST RESULTAT KMEANS                 %
% Affichage des frontières et des centres des superpixels %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Affichage des frontières/régions des superpixels
figure();
B=boundarymask(labels(:,:,29));
imshow(imoverlay(im(:,:,:,29),B,'red'));
title('Affichage des germes et régions');

% Affichage des centres des superpixels
figure();
imshow(im(:,:,:,29));
hold on;
title('Image avec centres');
hold on;
scatter(Centre(29,1:K_reel-1,4), Centre(29,1:K_reel-1,5), 'ro');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                       BINARISATION                      %
%         Binarise fond/forme sur les superpixels         %
%              Seuil appliqué sur les centres             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Seuils L, a et b
seuil_b = 115; % 115 c'est la fronntière entre le dino à 175 et le fond en deçà de 100
seuil_l = 0;
seuil_a = 0;
im_mask = zeros(nb_images, n,m); % Initialisation des images binarisée (36)

% Boucle de binarisation pour toutes les images, tous les superpixels
for img=1 : nb_images
    for l=1:K_reel
        for i=1:n
            for j=1:m
                if double(Centre(img,l,2))>seuil_a && double(Centre(img,l,3))>seuil_b && double(Centre(img,l,1))>seuil_l && labels(i,j,img)==l
                    im_mask(img,i,j)=1;
                end
            end
        end
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                 TEST RESULTAT BINARISATION              %
%          Affichage des images binarisées (masques)      %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure();
imshow(squeeze(im_mask(3,:,:))); % Image 3

figure();
imshow(squeeze(im_mask(9,:,:))); % Image 9

figure();
imshow(squeeze(im_mask(12,:,:))); % Image 12

figure();
imshow(squeeze(im_mask(24,:,:)));% Image 24

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%          ESTIMATION DES FONTIERES DU DINOSAURE          %
%               On applique bwtraceboundary               %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

mask = squeeze(im_mask(24,:,:)); % Passage de 3 dimensions à 2 (on enlève l'indice de l'image)
rows_white = any(mask, 2); % Vérifier pour chaque ligne si il y a au moins un pixel blanc
first_row = find(rows_white, 1, 'first'); % Première ligne contenant du blanc

cols_in_first_row = find(mask(first_row, :));
first_col = cols_in_first_row(1); % Premier pixel blanc dans cette ligne

row = first_row;
col = first_col;

boundary=bwtraceboundary(squeeze(im_mask(24,:,:)),[row,col],'N');

% Affichage de l'image 24 avec les fontières du dinosaure
figure();
imshow(squeeze(im_mask(24,:,:)));
hold on;
plot(boundary(:,2),boundary(:,1),'g','LineWidth',2);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%  ESTIMATION DES POINTS ET DE LA TOPOLOGIE DU SQUELETTE  %
%                   On applique Voronoï                   %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

voronoi_segments = cell(1, nb_images); % Initialiser une cellule pour stocker les segments par image

% Étape 3.2 : Estimation des points du squelette
for img = 1:nb_images

    % Trouver un point sur la frontière de l'objet
    mask = squeeze(im_mask(img,:,:));
    rows_white = any(mask, 2); % Lignes contenant au moins un pixel blanc
    first_row = find(rows_white, 1, 'first'); % Première ligne blanche
    
    cols_in_first_row = find(mask(first_row, :));
    first_col = cols_in_first_row(1); % Première colonne blanche

    row = first_row;
    col = first_col;

    if isempty(row) || isempty(col)
        continue; % Aucun pixel trouvé, on saute cette image
    end

    % Extraction de la frontière (contour fermé)
    boundary = bwtraceboundary(mask, [row, col], 'N');

    if isempty(boundary)
        continue; % Pas de frontière trouvée
    end

    % Sous-échantillonnage et nettoyage
    x = boundary(:,2);  % Colonnes
    y = boundary(:,1);  % Lignes

    step = 3;  % Sous-échantillonnage (1 point sur 3)
    x = x(1:step:end);
    y = y(1:step:end);

    % Supprimer les doublons
    [xy_unique, ~, ~] = unique([x y], 'rows');
    x = xy_unique(:,1);
    y = xy_unique(:,2);

    % Vérification : au moins 4 points requis pour un diagramme de Voronoï
    if size(x,1) < 4
        continue;
    end

    % Calcul du diagramme de Voronoï à partir des points de bord
    [V, C] = voronoin([x y]);

    % Étape 3.3 : Estimation de la topologie du squelette
    for i = 1:length(C)
        verts_idx = C{i};
        
        % Ignorer les cellules infinies (indice 1 dans voronoin)
        if any(verts_idx == 1)
            continue
        end

        verts = V(verts_idx, :);  % Sommets de la cellule courante

        % Générer les segments entre chaque paire consécutive de sommets
        for j = 1:size(verts,1)
            pt1 = verts(j,:);
            pt2 = verts(mod(j, size(verts,1)) + 1, :); % boucle circulaire
            cx1 = round(pt1(1)); cy1 = round(pt1(2));
            cx2 = round(pt2(1)); cy2 = round(pt2(2));

            % Vérification que les points sont dans les bornes de l'image
            if cx1 > 0 && cx1 <= m && cy1 > 0 && cy1 <= n && ...
               cx2 > 0 && cx2 <= m && cy2 > 0 && cy2 <= n

                % Vérification que les deux extrémités sont à l'intérieur de l'objet
                if mask(cy1, cx1) == 1 && mask(cy2, cx2) == 1
                    % Ajouter le segment interne au squelette
                    voronoi_segments{img}(end+1,:,:) = [pt1; pt2];
                end
            end
        end
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                TEST RESULTAT AXE MEDIAN                 %
%                                                         %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Affichage du masque
figure();
imshow(squeeze(im_mask(22,:,:)));
hold on;
title('Squelette: segments internes Voronoï');

% Affichage de l'axe médian
if ~isempty(voronoi_segments{22})
    for k = 1:size(voronoi_segments{22},1)
        seg = squeeze(voronoi_segments{22}(k,:,:));
        plot(seg(:,1), seg(:,2), 'm-');
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                AFFICHAGE 3D DU DINOSAURE                %
%                  PARTIE DÉJÀ ÉCRITE                     %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% chargement des points 2D suivis 
% pts de taille nb_points x (2 x nb_images)
% sur chaque ligne de pts 
% tous les appariements possibles pour un point 3D donne
% on affiche les coordonnees (xi,yi) de Pi dans les colonnes 2i-1 et 2i
% tout le reste vaut -1
pts = load('viff.xy');
% Chargement des matrices de projection
% Chaque P{i} contient la matrice de projection associee a l'image i 
% RAPPEL : P{i} est de taille 3 x 4
load dino_Ps;

% Reconstruction des points 3D
X = []; % Contient les coordonnees des points en 3D
color = []; % Contient la couleur associee
% Pour chaque couple de points apparies
for i = 1:size(pts,1)
    % Recuperation des ensembles de points apparies
    l = find(pts(i,1:2:end)~=-1);
    % Verification qu'il existe bien des points apparies dans cette image
    if size(l,2) > 1 & max(l)-min(l) > 1 & max(l)-min(l) < 36
        A = [];
        R = 0;
        G = 0;
        B = 0;
        % Pour chaque point recupere, calcul des coordonnees en 3D
        for j = l
            A = [A;P{j}(1,:)-pts(i,(j-1)*2+1)*P{j}(3,:);
            P{j}(2,:)-pts(i,(j-1)*2+2)*P{j}(3,:)];
            R = R + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),1,j));
            G = G + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),2,j));
            B = B + double(im(int16(pts(i,(j-1)*2+1)),int16(pts(i,(j-1)*2+2)),3,j));
        end;
        [U,S,V] = svd(A);
        X = [X V(:,end)/V(end,end)];
        color = [color [R/size(l,2);G/size(l,2);B/size(l,2)]];
    end;
end;
fprintf('Calcul des points 3D termine : %d points trouves. \n',size(X,2));

% Affichage du nuage de points 3D
figure;
hold on;
for i = 1:size(X,2)
    plot3(X(1,i),X(2,i),X(3,i),'.','col',color(:,i)/255);
end;
axis equal;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%              TETRAEDRISATION DE DELAUNAY                %
%                 On utilise DelaunayTri                  %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

T = DelaunayTri(X(1,:)', X(2,:)', X(3,:)');

% Afficher le maillage
fprintf('Tetraedrisation terminee : %d tetraedres trouves. \n',size(T,1));
figure;
tetramesh(T);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                CALCUL DES BARYCENTRES                   %
%                                                         %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Matrice des poids
poids = [0.25, 0.25, 0.25, 0.25; % Barycentre du tétraèdre
        10, 1, 1, 1; % Sommet 1
        1, 10, 1, 1; % Sommet 2
        1, 1, 10, 1; % Sommet 3
        1, 1, 1, 10]; % Sommet 4

nb_barycentres = size(poids,1); % Nombre de barycentres

% Initialisation de la matrice des barycentres de chaque tetraèdre
C_g = zeros(3, size(T,1), nb_barycentres); % [x;y;z] x nb_tetra x nb_bary

% On calcule C_g en fonction de la matrice poids
for i = 1:size(T,1)
    sommets = X(1:3, T(i,:)); % 3x4
    for k = 1:nb_barycentres
        C_g(:,i,k) = sommets * poids(k,:)';
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%     VERIFICATION DE LA PROJECTION DES BARYCENTRES       %
% Projection des barycentres selon la matrice de proj. P  %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%         A DECOMMENTER POUR VERIFICATION          %%%%
%%%%  A RE-COMMENTER UNE FOIS LA VERIFICATION FAITE   %%%%

% Visualisation pour vérifier le bon calcul des barycentres
% 
% for i = 1:nb_images
%     for k = 1:nb_barycentres
%         % Projeter tous les barycentres du type k dans l’image i
%         o = P{i} * [C_g(:,:,k); ones(1, size(C_g,2))];  % homogénéisation
%         o = o ./ repmat(o(3,:), 3, 1); % normalisation
% 
%         % Affichage du masque avec superposition des projections
%         imshow(squeeze(im_mask(i,:,:)));
%         hold on;
%         plot(o(2,:), o(1,:), 'rx'); % points rouges croix
%         title(sprintf('Image %d – barycentres %d', i, k));
%         pause; % attendre touche clavier
%         close;
%     end
% end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%          ELIMINATION DES TETRAÈDRES SUPERFLUS           %
% On enlève les barycentres projetés en dehors du masque  %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Copier la triangulation pour pouvoir filtrer
tri = T.Triangulation;               % Triangulation complète issue de delaunay
is_valid = true(size(T,1), 1);       % Initialisation : tous les tétraèdres sont valides par défaut

% Parcourir toutes les images et tous les types de barycentres
for i = 1 : nb_images
   for k = 1 : nb_barycentres

       % Projection des barycentres dans l'image i
       o = P{i} * [C_g(:,:,k); ones(1, size(C_g,2))];  % Projection homogène
       o = o ./ o(3,:);  % Normalisation par la coordonnée homogène

       % Coordonnées image (u = colonne, v = ligne)
       u = round(o(2,:));
       v = round(o(1,:));

       % Extraction du masque binaire correspondant à l'image i
       mask_i = squeeze(im_mask(i,:,:));  % Masque 2D

       % Vérification que les points projetés tombent bien dans l'image
       inside = u >= 1 & v >= 1 & u <= size(mask_i,2) & v <= size(mask_i,1);
       u_inside = u(inside);
       v_inside = v(inside);

       % Associer chaque barycentre projeté (valide) à un indice de tétraèdre
       tetra_indices = find(inside);

       % Parcourir les barycentres projetés valides
       for idxtetra = 1:length(u_inside)
           t = tetra_indices(idxtetra);  % Numéro du tétraèdre correspondant

           % Tester si le barycentre projeté est dans la région blanche (objet)
           if mask_i(v_inside(idxtetra), u_inside(idxtetra)) == 0
               is_valid(t) = false;  % Invalider le tétraèdre s'il tombe hors de l'objet
           end
       end
   end
end

% Supprimer les tétraèdres invalides (hors objet 3D)
T_filtre = tri(is_valid, :);  % Filtrage de la triangulation
fprintf('Tetraedrisation filtrée : %d tetraedres valides restants.\n', size(T_filtre,1));

% Visualisation des tétraèdres internes conservés
figure;
trisurf(T_filtre, X(1,:), X(2,:), X(3,:),'FaceAlpha', 0.3, 'EdgeColor', 'k');
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Tétraèdres internes (filtrés)');

% Sauvegarde des données utiles pour la suite
save donnees T_filtre X

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%            MAILLAGE FINAL DU DINOSAURE EN 3D            %
%        On appelle la fonction extraire_maillage         %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% Charger les données sauvegardées
load donnees;  % Charge T_filtre et X

% Calcul du maillage surfacique
FACES = extraire_maillage(T_filtre, X);
