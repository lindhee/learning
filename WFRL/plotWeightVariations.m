% Plot how the resulting weights vary when we collect more experience from the extreme game cases
% Magnus Lindhé, 2018

W_p=load('resulting_weights_padding.txt');
W_n=load('resulting_weights_no_padding.txt');

std_p = reshape(sqrt(var(W_p)),3,11);
std_n = reshape(sqrt(var(W_n)),3,11);

figure(1);
clf;
subplot(3,1,1);
plot(1:11,std_n(1,:),1:11,std_p(1,:));
title('Go left');
xlabel('Game state');
ylabel('Weight stddev');
legend('No experience bias','Experience bias');
subplot(3,1,2);
plot(1:11,std_n(2,:),1:11,std_p(2,:));
title('Go straight');
xlabel('Game state');
ylabel('Weight stddev');
subplot(3,1,3);
plot(1:11,std_n(3,:),1:11,std_p(3,:));
title('Go right');
xlabel('Game state');
ylabel('Weight stddev');

