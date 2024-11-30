function z = amri_sig_r2z(r)
 
    % Fisher's r-to-z-transformatiom
    z =.5.*log((1+r)./(1-r));

end