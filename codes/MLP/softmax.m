function[y_softmax] = softmax(y)
    y_softmax = [];
    sum = 0;
    for i = 1:size(y,2)
        sum = sum + exp(y(1,i));
    end
    for i = 1:size(y,2)
        c = exp(y(1,i))/sum;
        y_softmax = [y_softmax,c]; 
    end
end