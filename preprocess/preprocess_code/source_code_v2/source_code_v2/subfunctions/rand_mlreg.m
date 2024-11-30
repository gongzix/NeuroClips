
function para = rand_mlreg(dimV, dimH, init, sigma)

if strcmpi(init,'normal')
    para.W = sigma*randn(dimV, dimH);
else
    para.W = (2*rand(dimV, dimH)-1)/(dimV+dimH);
end

para.b = zeros(1, dimH);

end

