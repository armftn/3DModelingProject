function [labels, distances, Centre, K_reel] = kmeans_segmentation(im, n ,m , K, nb_images,  Centre, S, facteur, total_iter)
% KMEANS_SEGMENTATION Effectue une segmentation k-means (SLIC simplifié) sur plusieurs images en Lab
%
% Entrées :
%   - im : images en espace sRGB (n x m x 3 x nb_images)
%   - n, m : dimensions de chaque image
%   - K : nombre de superpixels (centres)
%   - nb_images : nombre d'images à traiter
%   - Centre : tableau des centres initiaux [nb_images x K x 5]
%   - S : espacement entre les centres (≈ taille d’un superpixel)
%   - facteur : facteur de pondération spatiale (compactness)
%   - total_iter : nombre d’itérations de l’algorithme
%
% Sorties :
%   - labels : carte des labels de superpixels [n x m x nb_images]
%   - distances : distances minimales pour chaque pixel
%   - Centre : centres mis à jour
%   - K_reel : nombre réel de centres utilisés

    % Étape 1 : Initialisation des matrices
    labels = zeros(n, m, nb_images);
    distances = inf(n, m, nb_images);
    n_part = floor(n/S); % nombre de lignes de centres
    m_part = floor(m/S); % nombre de colonnes de centres

    % Étape 2 : Parcourir toutes les images
    for img = 1:nb_images
        % Conversion RGB -> Lab
        cform = makecform('srgb2lab');
        lab = applycform(im(:,:,:,img), cform);

        % Placement initial des centres
        K_reel = 1;
        for i = 0:n_part-1
            for j = 0:m_part-1
                y = floor(S/2 + i * S);
                x = floor(S/2 + j * S);
                L = double(lab(y,x,1));
                a = double(lab(y,x,2));
                b = double(lab(y,x,3));
                Centre(img,K_reel,:) = [L, a, b, x, y];
                K_reel = K_reel + 1;
            end
        end

        % Étape 3 : Itérations de k-means
        for total = 1:total_iter

            % Assignation des pixels aux centres les plus proches
            for k = 1:K
                c_x = Centre(img, k, 4);
                c_y = Centre(img, k, 5);

                % Fenêtre de recherche locale autour du centre
                x_min = max(round(c_x - S), 1);
                x_max = min(round(c_x + S), m);
                y_min = max(round(c_y - S), 1);
                y_max = min(round(c_y + S), n);

                for i = y_min:y_max
                    for j = x_min:x_max
                        % Distance en Lab
                        dL = double(lab(i,j,1)) - double(Centre(img, k, 1));
                        da = double(lab(i,j,2)) - double(Centre(img, k, 2));
                        db = double(lab(i,j,3)) - double(Centre(img, k, 3));
                        d_lab = sqrt(dL^2 + da^2 + db^2);

                        % Distance spatiale
                        d_xy = sqrt((i - c_y)^2 + (j - c_x)^2);

                        % Distance totale avec pondération spatiale
                        Ds = d_lab + (facteur/S) * d_xy;

                        % Mise à jour si meilleure distance trouvée
                        if Ds < distances(i,j,img)
                            distances(i,j,img) = Ds;
                            labels(i,j,img) = k;
                        end
                    end
                end
            end

            % Mise à jour des centres par barycentre spatial
            for l = 1:K
                sumx = 0; sumy = 0; count = 0;
                for i = 1:n
                    for j = 1:m
                        if labels(i,j,img) == l
                            sumx = sumx + j;
                            sumy = sumy + i;
                            count = count + 1;
                        end
                    end
                end
                % Mise à jour si le centre a été assigné
                if count > 0
                    nouveau_centrex = sumx / count;
                    nouveau_centrey = sumy / count;
                    Centre(img, l, 4) = nouveau_centrex;
                    Centre(img, l, 5) = nouveau_centrey;
                end
            end
        end
    end
end
