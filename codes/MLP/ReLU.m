function[z_h]= ReLU(x,w)
    z_h = x*w';
    if z_h<0
        z_h=0;
    end
end