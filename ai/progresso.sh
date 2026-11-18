#!/bin/zsh
# Mostra l'avanzamento del download del modello Hunyuan3D-2mini (si aggiorna ogni 5 secondi).
F="${0:A:h}/modelli/tencent/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini/model.fp16.safetensors"
TOT=3819958234
prev=$(stat -f %z "$F")
while true; do
  sleep 5
  cur=$(stat -f %z "$F")
  v=$(( (cur - prev) / 5 )); prev=$cur
  pct=$(( cur * 100 / TOT ))
  n=$(( pct * 30 / 100 ))
  bar=$(printf '%*s' $n '' | tr ' ' '#')$(printf '%*s' $((30 - n)) '' | tr ' ' '.')
  if (( v > 0 )); then eta="mancano ~$(( (TOT - cur) / v / 60 )) min"; else eta=""; fi
  printf '\r[%s] %3d%%  %4d / 3643 MB  %4d KB/s  %s   ' "$bar" $pct $((cur / 1048576)) $((v / 1024)) "$eta"
  if (( cur >= TOT )); then printf '\nDownload finito!\n'; break; fi
  if ! pgrep -f "curl -sSL -C - --retry" >/dev/null; then printf '\nIl download si e fermato (lo riprendo io).\n'; break; fi
done
