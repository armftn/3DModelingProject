function FACES = extraire_maillage(T_filtre, X)
% EXTRAIRE_MAILLAGE Extrait les faces de surface à partir de la tétraédralisation filtrée
% 
% Entrée :
%   T_filtre : matrice des tétraèdres (n_tetra x 4)
%   X        : (optionnel) coordonnées des points 3D (3 x n_points)
%
% Sortie :
%   FACES    : liste des faces visibles (n_faces x 3)

    % Étape 1 : Générer toutes les faces
    faces = [ ...
        T_filtre(:, [1 2 3]);
        T_filtre(:, [1 2 4]);
        T_filtre(:, [1 3 4]);
        T_filtre(:, [2 3 4])];

    % Étape 2 : Tri local des sommets de chaque face (pour ranger par ordre croissant par ligne)
    faces = sort(faces, 2);

    % Étape 3 : Tri global (ligne par ligne) pour que les doublons se suivent
    faces = sortrows(faces);

    % Étape 4 : Détection des doublons (faces internes)
    N = size(faces,1); % nombre de faces générés
    mask = false(N,1); % true pour les lignes à ignorer

    for i = 1:N-1
        if isequal(faces(i,:), faces(i+1,:)) % On compare chaque face avec la suivante
            mask(i) = true;
            mask(i+1) = true;
        end
    end

    % Étape 5 : Garder uniquement les faces visibles
    FACES = faces(~mask, :);

    fprintf('Calcul du maillage final terminé : %d faces.\n', size(FACES,1));

    % Affichage graphique 3D
    figure;
    hold on;
    trisurf(FACES, X(1,:), X(2,:), X(3,:), ...
        'FaceColor', 'none', 'EdgeColor', 'k', 'FaceAlpha', 0.5);
    axis equal;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Maillage surfacique final');
    hold off;
end
