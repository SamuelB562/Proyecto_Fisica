"""Experimento independiente para la versión proy entrega final corregido.
Copiar a la raíz del proyecto. No modifica los módulos originales.
Reinicio MANUAL de cada sesión: la interfaz disponible no ofrece reset remoto.
"""
import argparse
import csv
import json
import math
import sys
import time
import traceback
import faulthandler
from pathlib import Path
from dataclasses import asdict, replace
from datetime import datetime
import numpy as np
DT = .02
MUESTREO = .2
CAMPOS = ['time_s','distance_m','x','y','heading_rad','speed_kmh','speed_target_kmh',
          'lateral_error_m','center_offset_m','outside_center','gas_cmd','brake_cmd',
          'steer_cmd_rad','gas_obs','brake_obs','steer_obs_raw','accG_0','accG_1','accG_2',
          'ax_model_m_s2','ay_model_m_s2','fuel','tyre_temp_0','tyre_temp_1','tyre_temp_2',
          'tyre_temp_3','tyre_wear_0','tyre_wear_1','tyre_wear_2','tyre_wear_3','packet_id','status']
def log(mensaje):
    print(mensaje, flush=True)
def guardar_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x)), encoding='utf-8')
def guardar_csv(path, rows, campos):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); w.writerows(rows)
def proyectar(xy, p, s, anterior):
    # Ventana de búsqueda local: evita saltar a otra rama de la pista.
    ids = np.flatnonzero((s[:-1] >= anterior-30) & (s[:-1] <= anterior+125))
    if not len(ids): raise ValueError('Proyección sin segmentos locales')
    a = p[ids]; d = p[ids+1]-a
    u = np.clip(np.sum((xy-a)*d,axis=1)/np.sum(d*d,axis=1),0,1)
    pies = a+u[:,None]*d
    j = np.argmin(np.sum((xy-pies)**2,axis=1)); i = ids[j]
    n = np.array([-d[j,1],d[j,0]])/np.linalg.norm(d[j])
    return float(s[i]+u[j]*(s[i+1]-s[i])), float((xy-pies[j])@n)
def leer_csv(path):
    return np.genfromtxt(path,delimiter=',',names=True,dtype=float,encoding='utf-8')
def preparar(out, e0, p, mod):
    import modelado_fisico as mf
    from estimaciones import evaluar_trayectoria
    nombre,variante,geo = mod.construir_geometria()
    c,iz,de = mod.orientar(geo,e0)
    c,iz,de = mod.suavizar_geometria(c,iz,de)
    c,iz,de,L,inicio_gestor = mod.preparar_vuelta(c,iz,de,np.array([e0.x,e0.y]),margen_atras_m=200,margen_despues_m=650)
    keep=np.r_[True,np.linalg.norm(np.diff(c,axis=0),axis=1)>1e-5]
    c,iz,de=c[keep],iz[keep],de[keep]; sc=mod.estaciones(c)
    inicio,_=mod.progreso([e0.x,e0.y],c,sc,anterior=inicio_gestor)
    if abs(inicio-inicio_gestor)>5 or inicio+L>sc[-1]:
        raise ValueError(f"Estación inicial incompatible: {inicio:.3f} m, gestor={inicio_gestor:.3f} m, longitud={L:.3f} m")
    log(f"[GEOMETRIA] inicio={inicio:.3f} m | vuelta={L:.3f} m | extensión={sc[-1]:.3f} m")
    # Insertar exactamente la estación inicial si no existe. El perfil de
    # velocidad arranca allí, no 200 m antes. Geometría anterior: solo contexto.
    j=int(np.searchsorted(sc,inicio))
    if 0<j<len(sc) and abs(sc[j]-inicio)>1e-6 and abs(sc[j-1]-inicio)>1e-6:
        q=(inicio-sc[j-1])/(sc[j]-sc[j-1])
        c,iz,de=[np.insert(a,j,a[j-1]+q*(a[j]-a[j-1]),axis=0) for a in (c,iz,de)]
    sc=mod.estaciones(c); j=int(np.argmin(abs(sc-inicio))); inicio=float(sc[j]); meta=inicio+L
    veh=mf.Vehiculo(masa_kg=p['masa_kg'],potencia_util_w=p['potencia_maxima_telemetria'],distancia_ejes_m=p['distancia_ejes_m'],relacion_direccion=p['direccion_steer_ratio'],aceleracion_max_m_s2=p['aceleracion_max_m_s2'],frenado_max_m_s2=p['frenado_max_m_s2'],coef_friccion=p['coef_friccion_modelo'])
    # Mismo evaluador y optimizador original, con condición inicial corregida.
    # Los ceros del prefijo no se integran ni se ejecutan.
    def evaluar_desde_salida(ruta, veh, v0):
        try:
            ev=evaluar_trayectoria(ruta[j:],veh,velocidad_entrada_m_s=0.,tope_velocidad_m_s=mod.TOPE_EVALUADOR_M_S)
            k=mf.curvaturas_de_puntos(ruta)
            return replace(ev,velocidades_m_s=np.r_[np.zeros(j),ev.velocidades_m_s],curvaturas_menos1=k,direcciones_rad=mf.curvatura_a_direccion(k,veh))
        except (ValueError,ArithmeticError): return None
    original=mod.evaluar
    # Capturar decisiones aceptadas sin reimplementar la búsqueda de ápices.
    # El trazador solo observa el marco de optimizar; no modifica su lógica.
    apices={}; ultimo=0
    def trace(frame,event,arg):
        nonlocal ultimo
        if frame.f_code is not mod.optimizar.__code__: return None
        if event=='line':
            z=frame.f_locals; count=z.get('mejoras',0)
            if count>ultimo and z.get('etiqueta') is not None:
                tipo,fi,fe,ext=z['etiqueta']; idx=z['idx']; numero=z['numero']
                sa=float(sc[idx]+{'temprano':-20,'medio':0,'tardio':20}[tipo])
                apices[numero]=dict(curva=numero,estacion_m=sa,tipo=tipo,interior=fi,exterior=fe,extension_m=ext)
                ultimo=count
        return trace
    mod.evaluar=evaluar_desde_salida
    try:
        ec=evaluar_desde_salida(c,veh,0)
        if ec is None: raise ValueError('Centro no evaluable desde reposo')
        oldtrace=sys.gettrace(); sys.settrace(trace)
        try: r,eo=mod.optimizar(c,iz,de,veh,0.,inicio,meta)
        finally: sys.settrace(oldtrace)
    finally: mod.evaluar=original
    if not np.all(np.isfinite(eo.velocidades_m_s)) or not np.all(np.isfinite(ec.velocidades_m_s)):
        raise ValueError('Perfil de velocidad no finito')
    if ec.velocidades_m_s[j+1] <= 0 or eo.velocidades_m_s[j+1] <= 0:
        log('[AVISO] Algún perfil tiene velocidad cero en el siguiente punto; revisar diagnóstico de arranque.')
    log(f'[PLAN] siguiente objetivo central={ec.velocidades_m_s[j+1]*3.6:.2f} km/h | optimizada={eo.velocidades_m_s[j+1]*3.6:.2f} km/h')
    if np.max(np.linalg.norm(r-c,axis=1))<1e-6:
        log('[AVISO] Optimización no produjo una ruta distinta. Revisar curvas y parámetros antes de validar una ganancia.')
    guardar_json(out/'parametros.json',{'lectura_AC':p,'vehiculo':asdict(veh),'circuito':nombre,'variante':variante,'inicio':asdict(e0),'dt_control':DT,'dt_csv':MUESTREO,'tope_evaluador_m_s':mod.TOPE_EVALUADOR_M_S,'apices_ultima_decision':list(apices.values())})
    planes={}
    for nombre,ruta,ev in [('central',c,ec),('optimizada',r,eo)]:
        sr=mod.estaciones(ruta); t=mod.tiempo_vuelta_plan(ruta,ev,sc,inicio,meta)
        planes[nombre]=(ruta,ev,sr)
        # Estimación espacial para reproducir las gráficas sin ejecutar AC.
        ss=np.r_[inicio,sc[(sc>inicio)&(sc<meta)],meta]
        vv=np.interp(ss,sc,ev.velocidades_m_s); rr=np.interp(ss,sc,sr)
        tt=np.r_[0.,np.cumsum(2*np.diff(rr)/np.maximum(vv[:-1]+vv[1:],1e-12))]
        guardar_csv(out/f'{nombre}_estimacion.csv',[dict(distance_m=float(a-inicio),time_s=float(b),speed_kmh=float(v*3.6),x=float(x),y=float(y),curvature_m_inv=float(k),lateral_error_m=0.) for a,b,v,x,y,k in zip(ss,tt,vv,np.interp(ss,sc,ruta[:,0]),np.interp(ss,sc,ruta[:,1]),np.interp(ss,sc,ev.curvaturas_menos1))],['distance_m','time_s','speed_kmh','x','y','curvature_m_inv','lateral_error_m'])
        # Guardar la geometría completa (incluidos márgenes) para futuros análisis.
        np.savez_compressed(out/f'plan_{nombre}.npz',centro=c,izquierdo=iz,derecho=de,ruta=ruta,sc=sc,sr=sr,v=ev.velocidades_m_s,k=ev.curvaturas_menos1,delta=ev.direcciones_rad,inicio=inicio,meta=meta,tiempo=t)
    return c,iz,de,sc,inicio,meta,veh,planes
def ensayo(path, plan, geom, e0, mod, live=None, timeout=450):
    import modelado_fisico as mf
    c,iz,de,sc,inicio,meta,veh=geom
    ruta,ev,sr=plan
    if live: log("[INFO] Inicializando agente conductor")
    agente=mod.AgenteConductor(veh,periodo_control_s=DT)
    agente.establecer_trayectoria(ruta)
    if live: log("[OK] Trayectoria cargada; preparando CSV y bucle de control")
    sref=float(sr[int(np.argmin(abs(sc-inicio)))]); scprev=inicio; srprev=sref
    normales=mf.normales_izquierdas(c); ancho_i=np.sum((iz-c)*normales,axis=1); ancho_d=-np.sum((de-c)*normales,axis=1)
    t=0.; prox=0.; reloj=time.monotonic(); estado=e0; prev_t=0.; prev_s=inicio; last_packet=None; last_new=reloj
    siguiente_log=0.
    ultimo_avance_t=0.; mejor_avance=0.; aviso_inmovil=False
    resultado={'completada':False,'tiempo_s':None,'motivo':'timeout','distancia_m':0.,'fuera_centro':False}
    # Un CSV crudo por ensayo y otro a 0.2 s. No sobrescribir ensayos previos.
    with path.open('w',newline='',encoding='utf-8') as f, path.with_name(path.stem+'_raw.csv').open('w',newline='',encoding='utf-8') as raw:
        w=csv.DictWriter(f,fieldnames=CAMPOS); wr=csv.DictWriter(raw,fieldnames=CAMPOS); w.writeheader();wr.writeheader()
        try:
            if live: log("[INFO] Iniciando conducción; leyendo primer paquete")
            while t<=timeout:
                tic=time.monotonic(); obs={}
                if live:
                    fis,graf,mando=live; F=mod.leer(fis,mod.Physics); G=mod.leer(graf,mod.Graphics)
                    t=tic-reloj; estado=mod.convertir_estado(F,G)
                    if last_packet is None: log(f"[OK] Primer paquete: {F.packetId}; estado AC={G.status}; velocidad={F.speedKmh:.2f} km/h")
                    if int(F.packetId)!=last_packet: last_new=tic;last_packet=int(F.packetId)
                    if G.status!=2 or G.isInPit or tic-last_new>1.: raise RuntimeError('Sesión no LIVE, boxes o telemetría vencida')
                    obs=dict(gas_obs=F.gas,brake_obs=F.brake,steer_obs_raw=F.steerAngle,accG_0=F.accG[0],accG_1=F.accG[1],accG_2=F.accG[2],fuel=F.fuel,packet_id=F.packetId,status=G.status)
                    obs.update({f'tyre_temp_{i}':F.tyreCoreTemperature[i] for i in range(4)});obs.update({f'tyre_wear_{i}':F.tyreWear[i] for i in range(4)})
                xy=np.array([estado.x,estado.y]); s,off=proyectar(xy,c,sc,scprev); q,err=proyectar(xy,ruta,sr,srprev)
                scprev=max(scprev,s);srprev=max(srprev,q)
                fuera=bool(off>np.interp(s,sc,ancho_i) or off<-np.interp(s,sc,ancho_d))
                resultado['fuera_centro'] |= fuera;resultado['distancia_m']=max(0.,s-inicio)
                if resultado['distancia_m']>mejor_avance+0.25:
                    mejor_avance=resultado['distancia_m']; ultimo_avance_t=t; aviso_inmovil=False
                fin=s>=meta
                # El perfil empieza en reposo. Si la proyección queda antes de la
                # salida (o unos ulp por debajo), la orden debe consultar el
                # segmento de salida, no el prefijo de ceros de contexto.
                # No se modifica la posición medida ni el perfil físico.
                q_control=max(q, np.nextafter(sref, np.inf))
                objetivo=mod.controlar(agente,ev,sr,q_control,t)
                informe=agente.actualizar(t,estado); ctl=informe.controles
                # Adaptador físico explícito: pedal × capacidad, limitado por
                # potencia y círculo de fricción. No es telemetría de AC.
                ax=ay=None
                if not live:
                    k=math.tan(ctl.direccion_rad)/veh.distancia_ejes_m
                    grip=float(mf.aceleracion_longitudinal_por_agarre(estado.velocidad_m_s,k,veh))
                    ax=float(np.clip(ctl.acelerador*mf.aceleracion_motor(estado.velocidad_m_s,veh)-ctl.freno*veh.frenado_max_m_s2,-grip,grip))
                    ay=estado.velocidad_m_s**2*math.tan(estado.delta_rad)/veh.distancia_ejes_m
                row=dict(time_s=t,distance_m=s-inicio,x=estado.x,y=estado.y,heading_rad=estado.psi_rad,speed_kmh=estado.velocidad_m_s*3.6,speed_target_kmh=objetivo*3.6,lateral_error_m=err,center_offset_m=off,outside_center=int(fuera),gas_cmd=ctl.acelerador,brake_cmd=ctl.freno,steer_cmd_rad=ctl.direccion_rad,ax_model_m_s2=ax,ay_model_m_s2=ay,**obs)
                wr.writerow(row)
                if t+1e-9>=prox or fin or fuera:
                    w.writerow(row);f.flush();raw.flush();prox=(math.floor((t+1e-9)/MUESTREO)+1)*MUESTREO
                if t-ultimo_avance_t>8.0 and estado.velocidad_m_s<0.5 and not aviso_inmovil:
                    log(f'[AVISO] Sin avance durante 8 s: v_obj={objetivo*3.6:.2f} km/h, gas={ctl.acelerador:.3f}, error={err:.2f} m')
                    aviso_inmovil=True
                if fuera or abs(off)>10:
                    resultado['motivo']='Punto de referencia fuera del corredor o desviación excesiva';break
                if fin:
                    frac=np.clip((meta-prev_s)/max(s-prev_s,1e-9),0,1)
                    resultado.update(completada=True,tiempo_s=float(prev_t+frac*(t-prev_t)),motivo='Meta');break
                prev_t,prev_s=t,s
                if live:
                    if siguiente_log == 0.: log("[INFO] Enviando primer comando vJoy")
                    mando.enviar(**mod.convertir_controles(ctl,veh))
                    if t >= siguiente_log:
                        log(f"[LIVE] t={t:.1f}s | s={s-inicio:.1f}m | v={estado.velocidad_m_s*3.6:.1f}/{objetivo*3.6:.1f} km/h | gas cmd/obs={ctl.acelerador:.2f}/{F.gas:.2f} | freno cmd/obs={ctl.freno:.2f}/{F.brake:.2f} | giro={math.degrees(ctl.direccion_rad):.1f}° | error={err:.2f}m | paquete={F.packetId}")
                        siguiente_log=t+1.
                    time.sleep(max(0,DT-(time.monotonic()-tic)))
                else:
                    estado=mf.paso_bicicleta(estado,ax,ctl.direccion_rad,DT,veh);t+=DT
        except KeyboardInterrupt: resultado['motivo']='Interrumpido';raise
        except Exception as ex:
            resultado['motivo']=f'{type(ex).__name__}: {ex}'
            log('[ERROR] '+resultado['motivo']); traceback.print_exc()
        finally:
            guardar_json(path.with_suffix('.json'),resultado)
            log(f'[FIN] {path.name}: {resultado}')
    return resultado
def graficas(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    resumen=[]
    for tipo in ('central','optimizada'):
        p=np.load(out/f'plan_{tipo}.npz'); s=p['sc']-p['inicio']; L=float(p['meta']-p['inicio']); mask=(s>=0)&(s<=L); grid=np.linspace(0,L,1001)
        def data(path): return np.atleast_1d(leer_csv(path))
        def interp(a,key):
            if len(a)<2:return np.full_like(grid,np.nan)
            x=a['distance_m']; keep=np.r_[True,np.diff(np.maximum.accumulate(x))>1e-8];x=np.maximum.accumulate(x)[keep];y=a[key][keep]
            return np.interp(grid,x,y,left=np.nan,right=np.nan)
        bpath=out/f'{tipo}_bicicleta.csv'; b=data(bpath) if bpath.exists() else None
        acpaths=sorted(out.glob(f'{tipo}_ac_[0-9][0-9].csv')); good=[data(q) for q in acpaths if json.loads(q.with_suffix('.json').read_text())['completada']]
        for key,label,ref in [('speed_kmh','Velocidad (km/h)',np.interp(grid,s,p['v']*3.6)),('lateral_error_m','Desviación lateral (m; + izquierda)',np.zeros_like(grid))]:
            fig,ax=plt.subplots();ax.plot(grid,ref,label='Estimación / referencia')
            if b is not None:ax.plot(grid,interp(b,key),label='Bicicleta + conductor')
            if good:
                a=np.array([interp(z,key) for z in good]);n=np.sum(np.isfinite(a),axis=0);mu=np.divide(np.nansum(a,axis=0),n,out=np.full_like(grid,np.nan),where=n>0)
                sd=np.sqrt(np.divide(np.nansum((a-mu)**2,axis=0),n-1,out=np.full_like(grid,np.nan),where=n>1));line=ax.plot(grid,mu,label=f'AC (media de {len(good)} completas)')[0];ax.fill_between(grid,mu-sd,mu+sd,alpha=.2,color=line.get_color())
            # Fallos aparecen como trazas parciales, nunca imputar el resto.
            for i,q in enumerate(acpaths):
                if not json.loads(q.with_suffix('.json').read_text())['completada']:
                    z=data(q);ax.plot(z['distance_m'],z[key],color='gray',alpha=.4,ls=':',label='AC fallida' if i==next(j for j,a in enumerate(acpaths) if not json.loads(a.with_suffix('.json').read_text())['completada']) else None)
            ax.set(xlabel='Estación central relativa (m)',ylabel=label,title=tipo);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/f'{tipo}_{key}.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots();
        for key in ('izquierdo','derecho'):ax.plot(*p[key][mask].T,color='gray',lw=1)
        ax.plot(*p['centro'][mask].T,ls='--',label='Centro');ax.plot(*p['ruta'][mask].T,label='Estimación')
        if b is not None:ax.plot(b['x'],b['y'],label='Bicicleta',alpha=.8)
        for i,q in enumerate(acpaths):
            z=data(q);ax.plot(z['x'],z['y'],alpha=.22,label='AC (ensayos)' if i==0 else None)
        if tipo=='optimizada':
            ap=json.loads((out/'parametros.json').read_text())['apices_ultima_decision'];a=np.array([x['estacion_m'] for x in ap]);
            if len(a):ax.scatter(np.interp(a,p['sc'],p['ruta'][:,0]),np.interp(a,p['sc'],p['ruta'][:,1]),s=15,label='Últimas decisiones de ápice')
        ax.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(out/f'{tipo}_mapa.png',dpi=180);plt.close(fig)
        # Mapas de desviación por vuelta, incluyendo parciales y bicicleta.
        for q in ([bpath] if b is not None else [])+acpaths:
            z=data(q);xy=np.c_[z['x'],z['y']];fig,ax=plt.subplots();ax.plot(*p['ruta'][mask].T,'k--',lw=.8,label='Referencia');ax.plot(*p['izquierdo'][mask].T,color='gray',lw=.6);ax.plot(*p['derecho'][mask].T,color='gray',lw=.6)
            if len(xy)>1:
                coll=LineCollection(np.stack([xy[:-1],xy[1:]],axis=1),cmap='viridis',norm=Normalize(0,2),linewidth=2);coll.set_array(np.abs(z['lateral_error_m'][:-1]));ax.add_collection(coll);fig.colorbar(coll,ax=ax,label='|Error lateral| (m; escala saturada a 2 m)')
            ax.autoscale();ax.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(out/f'{q.stem}_desviacion.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots();ax.axhline(float(p['tiempo']),label='Estimación');
        if b is not None:
            bj=json.loads(bpath.with_suffix('.json').read_text());
            if bj['completada']:ax.axhline(bj['tiempo_s'],ls='--',label='Bicicleta')
        for i,q in enumerate(acpaths,1):
            j=json.loads(q.with_suffix('.json').read_text());resumen.append(dict(trayectoria=tipo,ensayo=i,**j))
            if j['completada']:ax.scatter(i,j['tiempo_s'],color='tab:orange')
            else:ax.text(i,.03,'X',transform=ax.get_xaxis_transform(),color='red',ha='center')
        ax.set(xlabel='Ensayo (X = fallido)',ylabel='Tiempo de vuelta (s)',title=f'{tipo}: {len(good)}/{len(acpaths)} completas');ax.legend();fig.tight_layout();fig.savefig(out/f'{tipo}_tiempos.png',dpi=180);plt.close(fig)
    guardar_csv(out/'resumen_ensayos.csv',resumen,['trayectoria','ensayo','completada','tiempo_s','motivo','distancia_m','fuera_centro'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--vueltas',type=int,default=10);ap.add_argument('--solo-graficas',type=Path);ap.add_argument('--salida',type=Path);ap.add_argument('--timeout',type=float,default=450);args=ap.parse_args()
    if args.solo_graficas:graficas(args.solo_graficas);return
    if args.vueltas<1 or args.timeout<=0:ap.error('Valores positivos requeridos')
    import conducir_fisica_optimizado as mod
    out=args.salida or Path('experimentos')/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    fis=graf=mando=None
    try:
        input('AC LIVE, en pista, detenido y alineado; ENTER para capturar referencia: ')
        fis=mod.conectar('acpmf_physics',mod.Physics);graf=mod.conectar('acpmf_graphics',mod.Graphics)
        def estado_actual():
            F=mod.leer(fis,mod.Physics);G=mod.leer(graf,mod.Graphics)
            if G.status!=2 or G.isInPit or abs(F.speedKmh)>.1:raise RuntimeError('Se exige LIVE, fuera de boxes y detenido (<0.1 km/h)')
            return mod.convertir_estado(F,G),F,G
        e0,F,G=estado_actual();e0=replace(e0,velocidad_m_s=0.)
        p=mod.obtener_parametros();base_fuel=float(F.fuel);base_temp=np.array(F.tyreCoreTemperature);base_wear=np.array(F.tyreWear)
        c,iz,de,sc,inicio,meta,veh,planes=preparar(out,e0,p,mod);geom=(c,iz,de,sc,inicio,meta,veh)
        resultados_bici={}
        for tipo,plan in planes.items():
            resultados_bici[tipo]=ensayo(out/f'{tipo}_bicicleta.csv',plan,geom,e0,mod,timeout=args.timeout)
        if not all(x['completada'] for x in resultados_bici.values()):
            log('[AVISO] Bicicleta no completó todas las vueltas; se conservan los fallos y continúa la campaña AC.')
        if np.max(np.linalg.norm(planes['central'][0]-planes['optimizada'][0],axis=1))<1e-6:
            log('[AVISO] Rutas idénticas: se ejecutan ambas modalidades; no hay mejora geométrica prevista.')
        for n in range(1,args.vueltas+1):
            for tipo in (('central','optimizada') if n%2 else ('optimizada','central')):
                while True:
                    input(f'Ensayo {n}/{args.vueltas} {tipo}: reinicia sesión en Content Manager, mismas condiciones; ENTER: ')
                    try:
                        log('[INFO] Leyendo telemetría tras reinicio')
                        e,F,G=estado_actual()
                        log(f'[OK] Telemetría: paquete={F.packetId}, velocidad={F.speedKmh:.2f} km/h')
                        log('[INFO] Leyendo parámetros del vehículo (obtener_parametros)')
                        faulthandler.dump_traceback_later(15, repeat=True)
                        try: p2=mod.obtener_parametros()
                        finally: faulthandler.cancel_dump_traceback_later()
                        log('[OK] Parámetros leídos; verificando condiciones iniciales')
                        if any(p2.get(k)!=p.get(k) for k in ('automovil','circuito','compuesto')):raise ValueError('Vehículo/circuito/compuesto distintos')
                        if math.hypot(e.x-e0.x,e.y-e0.y)>.25 or abs(math.atan2(math.sin(e.psi_rad-e0.psi_rad),math.cos(e.psi_rad-e0.psi_rad)))>math.radians(2):raise ValueError('Salida distinta: tolerancia 0.25 m y 2°')
                        if abs(F.fuel-base_fuel)>.25 or np.max(abs(np.array(F.tyreCoreTemperature)-base_temp))>3 or np.max(abs(np.array(F.tyreWear)-base_wear))>.5:raise ValueError('Combustible, temperatura o desgaste inicial fuera de tolerancia')
                        log('[OK] Condiciones iniciales verificadas')
                        break
                    except Exception as ex:log(f'[RECHAZADO] {ex}')
                guardar_json(out/f'{tipo}_ac_{n:02d}_inicio.json',{'estado':asdict(e),'fuel':F.fuel,'temperatura':list(F.tyreCoreTemperature),'desgaste':list(F.tyreWear),'packet':F.packetId,'parametros':p2})
                # Una sola adquisición durante la campaña. Tras un fallo,
                # verificar manualmente el coche antes de intentar recuperar vJoy.
                while True:
                    try:
                        if mando is None:
                            log('[INFO] Adquiriendo vJoy')
                            faulthandler.dump_traceback_later(15, repeat=True)
                            try: mando=mod.ControladorVJoy()
                            finally: faulthandler.cancel_dump_traceback_later()
                        else:
                            log('[INFO] Verificando conexión vJoy existente')
                            mando.reposo()
                        log('[OK] vJoy listo para el ensayo')
                        break
                    except Exception as ex:
                        log(f'[ERROR vJoy] {type(ex).__name__}: {ex}')
                        mando=None
                        input('Detén el coche manualmente, comprueba vJoy y cierra otros emisores. ENTER para reintentar: ')
                        e,F,G=estado_actual()
                        if math.hypot(e.x-e0.x,e.y-e0.y)>.25 or abs(math.atan2(math.sin(e.psi_rad-e0.psi_rad),math.cos(e.psi_rad-e0.psi_rad)))>math.radians(2):
                            raise RuntimeError('Salida distinta tras recuperar vJoy; reinicia la campaña.')
                        if abs(F.fuel-base_fuel)>.25 or np.max(abs(np.array(F.tyreCoreTemperature)-base_temp))>3 or np.max(abs(np.array(F.tyreWear)-base_wear))>.5:
                            raise RuntimeError('Condiciones iniciales cambiaron durante la recuperación.')
                try:
                    res=ensayo(out/f'{tipo}_ac_{n:02d}.csv',planes[tipo],geom,e,mod,(fis,graf,mando),args.timeout)
                    log(str(res))
                finally:
                    log('[INFO] Frenando al finalizar el ensayo; conservando conexión vJoy')
                    try:
                        mod.detener(mando,fis)
                        log('[OK] Mandos en reposo; conexión conservada')
                    except Exception as ex:
                        log(f'[ERROR vJoy al detener] {type(ex).__name__}: {ex}')
                        guardar_json(out/f'{tipo}_ac_{n:02d}_incidencia_vjoy.json',{'error':type(ex).__name__,'mensaje':str(ex),'fase':'detener'})
                        mando=None
                        input('No se pudo confirmar el frenado virtual. Detén el coche manualmente antes de continuar. ENTER: ')
    finally:
        if mando is not None:
            try:
                mando.cerrar()
                log('[OK] Mandos finales en reposo')
            except Exception as ex:
                log(f'[AVISO] No se pudo neutralizar vJoy al salir: {type(ex).__name__}: {ex}. Comprueba el coche manualmente.')
        if fis is not None:fis.close()
        if graf is not None:graf.close()
        if (out/'plan_central.npz').exists() and (out/'plan_optimizada.npz').exists():graficas(out)
    print('Resultados:',out.resolve())
if __name__=='__main__':main()
