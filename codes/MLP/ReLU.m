function[z_h]= ReLU(x,w)
    z_h=max(0,x*w');
end