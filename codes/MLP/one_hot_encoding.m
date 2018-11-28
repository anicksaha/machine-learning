function[y_one_hot] = one_hot_encoding(label,n_classes)
    y_one_hot = zeros(1,n_classes);
    y_one_hot(1,label+1) = 1;
end