# codes_guidovia
I codici si riferiscono ai programmi utilizzati per l'analisi dati della relazione di laboratorio sull'esperimento della guidovia effettuato presso i Laboratori dell'Universita' degli Studi di Padova nell'a.a. 2024-2025.

distanzaCorta.py
Il codice permette di estrarre i dati dalle tabelle che riportano le misure raccolte. Contiene al suo interno il codice che interpreta e grafica correttamente i valori di velocita' e tempo riguardanti la 'misura corta' - come definita nella relazione.

distanzalunga.py
Similmente a quanto esegue lo script 'distanzaCorta.py', ma esegue il procedimento adatto per l'analisi delle 'misure lunghe'. (il codice sarebbe potuto essere organizzato in maniera piu' efficiente unendo i due file, ma per comodita' di scrittura si e' preferito optare per questa scelta)

plot_parabola.py
Questo codice replica quanto fatto dai codici precedenti ma con alcune leggere differenze. La principale sta nel grafico, e quindi nell'interpolazione, dei dati. Questo script relaziona posizione-tempo, generando un fit conico e non lineare. Questo codice (brutalmente) presenta l'analisi di entrambe le tecniche di raccolta dati, con una sezione di codice che, opportunamente commentata, permette di passare da un'analisi dati all'altra
