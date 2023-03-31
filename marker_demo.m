clear all;
close all;
clc; 

fid = fopen('RAW_20230330_132744073.txt');  %open file in binary mode
bytes = fread(fid, [1 Inf], 'uint8');        %read the whole lot as bytes
fclose(fid);
binstart = strfind(bytes, '202');    % Finding the MARKER indices
raw_data=[];

sz=size(binstart,2);
if sz==2     % For 2 Marker
    raw_data=[raw_data bytes(1:binstart(1)-1)];
    raw_data=[raw_data bytes(binstart(1)+19:binstart(2)-1)];
    raw_data=[raw_data bytes(binstart(2)+19:numel(bytes))];
end
if sz==4     % For 3 Marker
    raw_data=[raw_data typecast(bytes(1:binstart(1)-1), 'int8')];
    raw_data=[raw_data typecast(bytes(binstart(1)+19:binstart(2)-1), 'int8')];
    raw_data=[raw_data typecast(bytes(binstart(2)+19:binstart(3)-1), 'int8')];  
    raw_data=[raw_data typecast(bytes(binstart(3)+19:binstart(3)-1), 'int8')]; 
    raw_data=[raw_data bytes(binstart(4)+19:numel(bytes))];
end
if sz==6     % For 3 Marker
    raw_data=[raw_data typecast(bytes(1:binstart(1)-1), 'int8')];
    raw_data=[raw_data typecast(bytes(binstart(1)+19:binstart(2)-1), 'int8')];
    raw_data=[raw_data typecast(bytes(binstart(2)+19:binstart(3)-1), 'int8')];  
    raw_data=[raw_data typecast(bytes(binstart(3)+19:binstart(3)-1), 'int8')]; 
    raw_data=[raw_data typecast(bytes(binstart(4)+19:binstart(3)-1), 'int8')];  
    raw_data=[raw_data typecast(bytes(binstart(5)+19:binstart(3)-1), 'int8')];
    raw_data=[raw_data bytes(binstart(6)+19:numel(bytes))];
end
if sz==8     % For 4 Marker
    raw_data=[raw_data bytes(1:binstart(1)-1)];
    raw_data=[raw_data bytes(binstart(1)+19:binstart(2)-1)];
    raw_data=[raw_data bytes(binstart(2)+19:binstart(3)-1)];  
    raw_data=[raw_data bytes(binstart(3)+19:binstart(3)-1)]; 
    raw_data=[raw_data bytes(binstart(4)+19:binstart(3)-1)];  
    raw_data=[raw_data bytes(binstart(5)+19:binstart(3)-1)];
    raw_data=[raw_data bytes(binstart(6)+19:binstart(3)-1)];  
    raw_data=[raw_data bytes(binstart(7)+19:binstart(3)-1)];
    raw_data=[raw_data bytes(binstart(8)+19:numel(bytes))];
end
bytes=raw_data;
bytes2=raw_data;
% binstart = strfind(bytes, '$') + 1;          %find location of #!. skip these two characters
% text = char(bytes(1:binstart-1));
% bindata = typecast(bytes(binstart:numel(bytes)), 'int8');

% spo2=[]; ir_filtered=[];ir_raw=[];red_raw=[]; ecg_2=[]; ecg_1=[];ecg_3=[]; bpm=[]; axx=[];axy=[];axz=[];
% % bytes=bytes(1200001:end);
% for i=1:22:length(bytes)-22,   %#ok<NOCOL>
%     spo2=[spo2 bytes(i+11)];  %#ok<AGROW>
%     ecg_1=[ecg_1 256.0 *bytes(1 + i)+ 1.0 *bytes(i + 2)]; %#ok<AGROW>
%     ecg_2=[ecg_2 256.0 *bytes(3 + i)+ 1.0 *bytes(i + 4)]; %#ok<AGROW>
%     ecg_3=[ecg_3 256.0 *bytes(5 + i)+ 1.0 *bytes(i + 6)]; %#ok<AGROW>
%     ir_filtered=[ir_filtered 256.0 *bytes(7 + i)+ 1.0 *bytes(i + 8)];
% %     ir_filtered=[ir_filtered bitor(bitshift(bytes(7 + i), -8), bytes(i + 8))];
%     ir_raw=[ir_raw 65536.0* bytes(12 + i) + 256.0 * bytes(13 + i) + 1.0 * bytes(14 + i) ];
%     red_raw=[red_raw 65536.0*bytes(15 + i) + 256.0 *bytes(16 + i) + 1.0 *bytes(i + 17)];
% %     ir_raw=[ir_raw bitor(bitshift(bytes(12 + i), -16), bitor(bitshift(bytes(13 + i), -8), bytes(i + 14)))];
% %     red_raw=[red_raw bitor(bitshift(bytes(15 + i), -16), bitor(bitshift(bytes(16 + i), -8), bytes(i + 17)))];
%     bpm=[bpm 256.0 *bytes(i+9)+ 1.0 *bytes(i + 10)];
%     axx=[axx bytes(i+18)];
%     axy=[axy bytes(i+19)];
% %     axz=[axz bytes(i+20)];
% end

% fid = fopen('RAW_20230330_132744073.txt');  %first measurmenet
% fid=fopen('RAW_20230324_143312098.txt');  % 23feb23
% bytes = fread(fid, [1 Inf], 'uint8');        %read the whole lot as bytes
% fclose(fid);
proto1=[];
proto2=[];
spo2=[]; ir_filtered=[];ir_raw=[];red_raw=[]; ecg_2=[]; ecg_1=[];ecg_3=[]; bpm=[]; axx=[];axy=[];axz=[];
% bytes=bytes(1200001:end);
for i=1:28:length(bytes)-28,   %#ok<NOCOL>
    if(bytes(i+1)==50 && bytes(i+3)==36)
        proto1=[proto1 bytes(i+2)];
        spo2=[spo2 bytes(i+13)];  %#ok<AGROW>
        ecg_1=[ecg_1 256.0 *bytes(4 + i)+ 1.0 *bytes(i + 5)]; %#ok<AGROW>
        ecg_2=[ecg_2 256.0 *bytes(6 + i)+ 1.0 *bytes(i + 7)]; %#ok<AGROW>
        ecg_3=[ecg_3 256.0 *bytes(8 + i)+ 1.0 *bytes(i + 9)]; %#ok<AGROW>
        ir_filtered=[ir_filtered 256.0 *bytes(10 + i)+ 1.0 *bytes(i + 11)];
%     ir_filtered=[ir_filtered bitor(bitshift(bytes(7 + i), -8), bytes(i + 8))];
        ir_raw=[ir_raw 65536.0* bytes(15 + i) + 256.0 * bytes(16 + i) + 1.0 * bytes(17 + i)];
        red_raw=[red_raw 65536.0*bytes(18 + i) + 256.0 *bytes(19 + i) + 1.0 *bytes(i + 20)];
%     ir_raw=[ir_raw bitor(bitshift(bytes(12 + i), -16), bitor(bitshift(bytes(13 + i), -8), bytes(i + 14)))];
%     red_raw=[red_raw bitor(bitshift(bytes(15 + i), -16), bitor(bitshift(bytes(16 + i), -8), bytes(i + 17)))];
        bpm=[bpm 256.0 *bytes(i+12)+ 1.0 *bytes(i + 13)];
        axx=[axx 256.0 *bytes(i+21)+ 1.0 *bytes(i + 22)];
        axy=[axy 256.0 *bytes(i+23)+ 1.0 *bytes(i + 24)];
        axz=[axz 256.0 *bytes(i+25)+ 1.0 *bytes(i + 26)];               
    end
end

% fid2 = fopen('RAW_20230330_132744073.txt');  %first measurement
% fid2=fid;
% bytes2 = fread(fid2, [1 Inf], 'uint8');        %read the whole lot as bytes
% fclose(fid2);

spo22=[]; ir_filtered2=[];ir_raw2=[];red_raw2=[]; ecg_22=[]; ecg_12=[];
ecg_32=[]; bpm2=[]; axx2=[];axy2=[];axz2=[];
% bytes=bytes(1200001:end);
for i=1:28:length(bytes2)-28,   %#ok<NOCOL>
     if(bytes2(i+1)==51 && bytes2(i+3)==36)
        proto2=[proto2 bytes2(i+2)];
        spo2=[spo2 bytes2(i+13)];  %#ok<AGROW>
        ecg_1=[ecg_1 256.0 *bytes2(4 + i)+ 1.0 *bytes2(i + 5)]; %#ok<AGROW>
        ecg_2=[ecg_2 256.0 *bytes2(6 + i)+ 1.0 *bytes2(i + 7)]; %#ok<AGROW>
        ecg_3=[ecg_3 256.0 *bytes2(8 + i)+ 1.0 *bytes2(i + 9)]; %#ok<AGROW>
        ir_filtered=[ir_filtered 256.0 *bytes2(10 + i)+ 1.0 *bytes2(i + 11)];
%     ir_filtered=[ir_filtered bitor(bitshift(bytes(7 + i), -8), bytes(i + 8))];
        ir_raw=[ir_raw 65536.0* bytes2(15 + i) + 256.0 * bytes2(16 + i) + 1.0 * bytes2(17 + i)];
        red_raw=[red_raw 65536.0*bytes2(18 + i) + 256.0 *bytes2(19 + i) + 1.0 *bytes2(i + 20)];
%     ir_raw=[ir_raw bitor(bitshift(bytes(12 + i), -16), bitor(bitshift(bytes(13 + i), -8), bytes(i + 14)))];
%     red_raw=[red_raw bitor(bitshift(bytes(15 + i), -16), bitor(bitshift(bytes(16 + i), -8), bytes(i + 17)))];
        bpm=[bpm 256.0 *bytes2(i+12)+ 1.0 *bytes2(i + 13)];
        axx=[axx 256.0 *bytes2(i+21)+ 1.0 *bytes2(i + 22)];
        axy=[axy 256.0 *bytes2(i+23)+ 1.0 *bytes2(i + 24)];
        axz=[axz 256.0 *bytes2(i+25)+ 1.0 *bytes2(i + 26)];               
    end
end


figure;
L=length(ecg_2);
Fs=33;

% fc = 0.1;
% Wn = pi*fc/(2*Fs);
% n = 3;
% [b,a] = butter(n, Wn, 'low');
% filteredSignal = filter(b, a, red_raw);
% filteredSignalRed = filteredSignal - mean(filteredSignal); % Subtracting the mean to block DC Component

% plot(spo2, 'r'); hold on; plot(ecg_1,'b');hold on; plot(bpm,'g');
Y=fft(red_raw);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
plot(f,P1); 
title('Single-Sided Amplitude Spectrum of REDRAW');
xlabel('f (Hz)');
ylabel('|P1(f)|');
axis([-0 25 0 400]);
% figure; plot(ecg_1);hold on; plot(ecg_2);hold on; plot(ecg_3);
% figure; title('SPO2, BPM, FILT-IR');
% plot(spo2); hold on;plot(bpm); hold on; plot(ir_filtered);hold on; plot(ecg_2);
% legend('Spo2', 'bpm','filt-ir', 'ecg-2');
% 
% % ir_raw=ir_raw(1:size(t));
% % red_raw=red_raw(1:size(t));
if sz==2
figure;subplot 211; plot((0:length(ir_raw)-1)/67,ir_raw,'b'); legend('Raw-Ir');
hold on; stem((binstart(1)-1)/67/60,100000, 'go');hold on; stem((binstart(2)-1)/67/60,100000,'go');
subplot 212; plot((0:length(ir_raw)-1)/67,red_raw,'r');legend('Raw-Red');
end

if sz==4
figure;
subplot 211; plot((0:length(ir_raw)-1)/33,ir_raw,'b'); legend('Raw-Ir'); hold on; 
stem((binstart(1)-1)/33/60,100000, 'go');hold on; 
stem((binstart(2)-1)/33/60,100000,'go'); hold on; 
stem((binstart(3)-1)/33/60,100000, 'go'); hold on;
stem((binstart(4)-1)/33/60,100000, 'go'); hold on;
subplot 212; plot((0:length(ir_raw)-1)/33,red_raw,'r');legend('Raw-Red'); 
end
if sz==6
figure;
subplot 211; plot((0:length(ir_raw)-1)/33,ir_raw,'b'); legend('Raw-Ir'); hold on; 
stem((binstart(1)-1)/33/60,100000, 'go');hold on; 
stem((binstart(2)-1)/33/60,100000,'go'); hold on; 
stem((binstart(3)-1)/33/60,100000, 'go'); hold on;
stem((binstart(4)-1)/33/60,100000, 'go'); hold on;
stem((binstart(5)-1)/33/60,100000, 'go'); hold on;
stem((binstart(6)-1)/33/60,100000, 'go'); hold on;
subplot 212; plot((0:length(ir_raw)-1)/33,red_raw,'r');legend('Raw-Red'); 
end

if sz==8
figure;
subplot 211; plot((0:length(ir_raw)-1)/33,ir_raw,'b'); legend('Raw-Ir'); hold on; 
stem(((binstart(1)-1)/28)/66/4, 'go');hold on; 
stem(((binstart(2)-1)/28)/66/4,100000,'go'); hold on; 
stem(((binstart(3)-1)/28)/66/4,100000, 'go'); hold on;
stem(((binstart(4)-1)/28)/66/4,100000, 'go'); hold on;
stem(((binstart(5)-1)/28)/66/4,100000, 'go'); hold on;
stem(((binstart(6)-1)/28)/66/4,100000, 'go'); hold on;
stem(((binstart(7)-1)/28)/66/4,100000, 'go'); hold on;
stem(((binstart(8)-1)/28)/66/4,100000, 'go'); hold on;
subplot 212; plot((0:length(ir_raw)-1)/33,red_raw,'r');legend('Raw-Red'); 
end
% figure; subplot 311; plot(axx,'b');
% subplot 312; plot(axy,'r');
% subplot 313; plot(axz,'g');
% 
% 
% figure; title('SPO2, BPM, FILT-IR, RAW-IR, RAW-RED');
% plot(spo2(10000:10500)); hold on;plot(bpm(10000:10500)); hold on; plot(ir_filtered(10000:10500));
% hold on; plot(ir_raw(10000:10500)); hold on; plot(red_raw(10000:10500));
% legend('Spo2', 'bpm','filtIr','rawir','rawred');

%take the fourier transform
xdft = fft(red_raw);
xdft = xdft(1:length(red_raw)/2+1);
% create a frequency vector
freq = 0:Fs/length(red_raw):Fs/2;
%locs is now in frequency 
[pks,locs] = findpeaks(20*log10(abs(xdft)), "MinPeakProminence", 30);
%finds the peaks within the range of the fundamental frequency of the notes
indesiredrange = locs > 150 & locs < 500;
%gets the subsets within range
pks_subset = pks(indesiredrange);
locs_subset = locs(indesiredrange);
figure(1)
semilogx(freq,20*log10(abs(xdft)))
hold on
plot(freq(locs_subset), pks_subset, 'or')
hold off
xlabel('f(Hz)');
title('FFT of signal')
harmonic = [1:20];
%gets the locations in frequency
[peaks,freqs] = findpeaks(20*log10(abs(xdft)),freq);
harmB = 244.3*harmonic;
%finds the frequency peaks nearest to the calculated harmonics of note B(244.3hz)
vqB = interp1(freqs,peaks,harmB, 'nearest');
figure;
semilogx(freq,20*log10(abs(xdft)))
hold on
plot(harmB, vqB, 'or')
hold off
xlabel('f(Hz)');
title('FFT of signal');

