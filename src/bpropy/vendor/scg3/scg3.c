/******************************************************************************
*              Euristica per la risoluzione del Set Covering                  *
******************************************************************************/

#include "scg3.h"
#include <string.h>

#if !defined(log2)
#define log2(x) (log(x) / log(2))
#endif

#if !defined(max) && !defined(__cplusplus)
#define max(a,b)  (((a) > (b)) ? (a) : (b))
#endif

#if !defined(min) && !defined(__cplusplus)
#define min(a,b)  (((a) < (b)) ? (a) : (b))
#endif


/******************* PROTOTIPI (in ordine di apparizione) ********************/
#ifndef ZDD
static void LeggiIstruzioni(int argc, char* argv[]);
static void CaricaDati(char* Dati);
static void CaricaParametri(char* Parametri);
#endif
static void Alloca(void);
static void Inizializza(void);
static void Dealloca(void);

static float ran1(long* idum);

static int RowDomination(void);
#ifdef DOS
static enum bool RigaDomina(int i2, int i1);
#else
static int RigaDomina(int i2, int i1);
#endif
static int ColumnInclusion(void);
static int ColumnDomination(void);
static int ColDomination(void);
#ifdef DOS
static enum bool ColonnaDomina(int j1, int j2);
#else
static int ColonnaDomina(int j1, int j2);
#endif

static void EliminaRiga(int i);
static void EliminaColonna(int j);
static void ColonnaOttima(int j);

static void CopiaSoluzione(int* Sol1, double Val1, int* Sol2, double* Val2);
static void CompletaSoluzione(int* Sol, double* Val);

static void UpperBound(double *Cost, int Funz, int *Solution, double* UB);
static void UpperBoundRid(double *Cost, int Funz, int *Solution, double* UB);
static double Funz1(double Cost, int Num);
static double Funz2(double Cost, int Num);
static double Funz3(double Cost, int Num);
static double Funz4(double Cost, int Indice);
static void EliminaRidondanti(int *Solution);

static void DualAscent(double *U, double *LB);
static double minDA(double *U, int i);
static int NumeroColonne(const void *A, const void *B);

static void SolvePLLB(double *PS, int *X, double *CRP, double *LB);
static void SolveDLUB(double *DS, int *U, double *CRD, double *UB);

static void SubgradientAscent(double *PS, double *DS, double *CRP, double *CRD,
                              double *LUB, double *LLB, int IterSG);

static int LagrangeanPenalties(double *L_C, double LB, double UB);
static int LagrangeanRowSplitting(double *LC, double LB, double UB);
static int DualAscentPenalties(double *U, double UB);

static void AggiornaPeggiori(struct Blocco *Peggiori, double temp, int j);
static void AggiornaMigliori(struct Blocco *Migliori, double temp, int j);

static void SalvaRisultati(char *Risultati);

static void SaveProblem(void);             /* conserva il problema corrente */
static void RestoreProblem(void);       /* ripristina il problema originale */

/****************************************************************************/


/**************************** VARIABILI GLOBALI *****************************/


/************************** Matrice di copertura ****************************/
#ifdef ZDD
struct MatRC *Ar;                      /* colonne che coprono ciascuna riga */
struct MatCR *Ac;                      /* righe coperte da ciascuna colonna */
#else
static struct MatRC* Ar = NULL;        /* colonne che coprono ciascuna riga */
static struct MatCR* Ac = NULL;        /* righe coperte da ciascuna colonna */
#endif

static int RigheIniz;                        /* numero di righe iniziali    */
static int ColonneIniz;                      /* numero di colonne iniziali  */
static int NumRighe;                         /* numero corrente di righe    */
static int NumColonne;                       /* numero corrente di colonne  */

static int temprighe, tempcol;           /* Indici di percorrenza ausiliari,
                                            per le riduzioni */

/********************** Soluzione parziale corrente *************************/
static int colott = 0;/* Indice della prima colonna nella soluzione parziale */
static double OUB = 0.0;                  /* Costo della soluzione parziale */

static int *Sol = NULL;             /* miglior soluzione euristica corrente */
static double ZUB = DBL_MAX;
static double ZLB = 0.0;                    /* miglior lower bound corrente */

static int *SolLUB = NULL;                           /* soluzione euristica */
static double LUB = DBL_MAX;                       /* upper bound euristico */
static double LLB;               /* lower bound ottenuto col sottogradiente */

/* Moltiplicatori lagrangiani primali e duali                               */
static double *PS = NULL;
static double *DS = NULL;

/* Costi ridotti lagrangiani primali e duali                                */
static double *CRP = NULL;
static double *CRD = NULL;

static struct Blocco Migliori;            /* Insieme delle colonne migliori */
static struct Blocco Peggiori;            /* Insieme delle colonne peggiori */


/* Nomi dei file di ingresso e uscita */
static char Dati[256+1], Parametri[256+1], Risultati[256+1];

/* Tempo di esecuzione e file di LOG */
#ifdef DOS
static clock_t inizio, fine;
static double tempo;
#else
static long inizio, fine;
static double tempo;
#endif

#ifndef ZDD
static FILE *lg;
#endif

/* Informazioni sulle prestazioni */

static int EliElem = 0, EliCol = 0, EliRow = 0, OptCol = 0;
static int ColDom = 0, RowDom = 0, ColIncl = 0;
static int DAPenOpt = 0, DAPenDel = 0, LagrPenOpt = 0, LagrPenDel = 0;
static long int SubGrad = 0, DualAsc = 0;


/************** Parametri dell'algoritmo (e valori di default) **************/

static int MaxIter = 1;       /* Massimo numero di iterazioni del programma */
static double MaxTempo = 3600.0;

/* Colonne da eliminare perche' le riduzioni, rispettivamente interne ed
   esterne, procedano */
static int MinRid  = 2;
static int MinRid2 = 3;

static int IterSG1 = 10;
static int IterSG2 = 7;

static int DIVI_COL_CORR = 100;
static int MINI_COL_CORR = 2000;
int COL_PEN_DUALI = 100;

static double F0 = 0.005;
static int ALFA = 8;

static double PR1 = 0.34;
static double PR2 = 0.33;
static double PR3 = 0.33;

static double MIN_GAP = 0.1;


/***************************** ColumnDomination *****************************/
#ifdef DOS
static enum bool *Usata = NULL;
#else
static int *Usata = NULL;
#endif

/******************************** UpperBound ********************************/

static double (*(Funzioni)[4+1])(double,int); /* Criteri per sceglier la colonna */
static double* Log2 = NULL;
static double* cLog2 = NULL;

static int *cover = NULL;
static int *covermio = NULL;
static int *Agg = NULL;            /* Numero aggiornamenti su quella colonna */
static int *DaAgg = NULL;          /* Colonne i cui costi vanno riaggiornati */
static int NumAgg;
static struct DatiCol *data = NULL;

static int *Ridondanti = NULL;                         /* Colonne ridondanti */
static int NumRidond;


/******************************** DualAscent *********************************/

static int *N = NULL;


/**************************** SubgradientAscent *******************************/

static int *LX = NULL;                       /* soluzione lagrangiana primale */
static int *LU = NULL;                         /* soluzione lagrangiana duale */
static int *PG = NULL;                  /* vettore dei sottogradienti primale */
static int *DG = NULL;                    /* vettore dei sottogradienti duale */

static double *UZMax = NULL; /* moltiplicatori primali del miglior lower bound noto */
static int rotate = 4;

/*************************** DualAscentPenalities ****************************/

static double *UR = NULL;


/************************** LagrangeanRowSplitting ***************************/
// ATTENZIONE: Per renderlo dinamico, cercare in Inizializza la riga piu' lunga,
//             dato che ColNeg contiene gli indici delle colonne che coprono
//             la riga correntemente considerata.
static int ColNeg[10000];

/*********************** SaveProblem / RestoreProblem ************************/

static struct MatRC* SAr = NULL;        /* colonne che coprono ciascuna riga */
static struct MatCR* SAc = NULL;        /* righe coperte da ciascuna colonna */

static int Srighecorr;                         /* numero corrente di righe   */
static int Scolonnecorr;                       /* numero corrente di colonne */

static int Scolott = 0;
static double SOUB = 0.0;

static double *SPS = NULL;
static double *SDS = NULL;

static double *SCRP = NULL;
static double *SCRD = NULL;

static int SEliElem = 0, SEliCol = 0, SEliRow = 0, SOptCol = 0;
static int SColDom = 0, SLagrPenOpt = 0, SLagrPenDel = 0;
static int SDAPenOpt = 0, SDAPenDel = 0, SColIncl = 0, SRowDom = 0;
static long int SSubGrad = 0, SDualAsc = 0;



/************************** PROGRAMMA PRINCIPALE *****************************/

#ifdef ZDD
sm_row *SCG(initial_rows, initial_columns)
int initial_rows;                              /* numero di righe   iniziali */
int initial_columns;                           /* numero di colonne iniziali */
#else
int main(int argc, char *argv[])
#endif
{ int i, j, l;
  long dummy;
  float prob;
  int Iterazioni;                      /* Iterazione corrente dell'algoritmo */
  double Valore;
#ifdef ZDD
  sm_row* cover;

  RigheIniz = initial_rows;
  ColonneIniz = initial_columns;
#else
  LeggiIstruzioni(argc,argv);
  CaricaDati(Dati);
#endif

  /* Inizializza il generatore di numeri casuali in (0,1]. */
  /*  srand((unsigned) time(&t)); */
  dummy = - 3001;
  prob = ran1(&dummy);

#ifdef DOS
  inizio = clock();
#else
  inizio = util_cpu_time();
#endif

  /* Alloca e inizializza le variabili globali. */
  Alloca();
  Inizializza();
  printf("\n\n\nProblema %s: %d righe; %d colonne\n",Dati,NumRighe,NumColonne);

  /* PASSI 1,2,3: Riduzioni primarie */
  do
  { RowDomination();
    j = ColumnInclusion();
    j += ColumnDomination();
#ifdef UNIF
    // RIPROVARE
    j += ColDomination();
#endif
    printf("Reductions: (%d x %d)\n",NumRighe,NumColonne);
  } while (j >= MinRid);

  /* PASSO 4: cerca un upper bound con l'euristica completa di Balas-Ho. */
  ZUB = DBL_MAX;
  for (l = 4; l <= 4; l++)
  { UpperBound(/*NULL,*/CRP,l,SolLUB,&LUB);
    if (LUB < ZUB) CopiaSoluzione(SolLUB,LUB,Sol,&ZUB); };

#ifdef ZDD
  // RIPROVARE: L'uso di UpperBoundRid a volte peggiora a volte migliora.
  //            Per adesso la uso solo sulla quarta euristica e sotto Joke.
  UpperBoundRid(/*NULL,*/CRP,4,SolLUB,&LUB);
  if (LUB < ZUB) CopiaSoluzione(SolLUB,LUB,Sol,&ZUB);
#endif

  if (ZUB >= DBL_MAX)
  { printf("\nProblema insolubile!\n\a");
    exit(0); };

  CompletaSoluzione(Sol,&ZUB);
  printf("\nBH  = %ld, ",(long) ZUB);

  /* PASSO 5: Inizializza i moltiplicatori lagrangiani primali e duali. */
  for (j = 1; Sol[j] > 0; j++)
    DS[Sol[j]] = 1.0;

  DualAscent(PS,&ZLB);
  printf("DA = %ld\n",(long) ZLB);

  /* PASSO 6 : Sottogradiente primario */

  SubgradientAscent(PS,DS,CRP,CRD,&LUB,&LLB,IterSG1);
  printf("\nSA\nZUB = %ld, UB = %ld, LB = %ld\n",(long) ZUB, (long) LUB, (long) LLB);

  if (LLB < ZUB)
  { SaveProblem();

    /* Compie MaxIter tentativi oppure si ferma se scade il tempo. */
    for (Iterazioni = 1; (Iterazioni <= MaxIter) /*&& ((clock()-inizio)/CLK_TCK <= MaxTempo)*/; Iterazioni++)
    { printf("\nIterazione %d\n",Iterazioni);

      Migliori.num = Iterazioni;
/*      if (Iterazioni == 1)
        Migliori.num = 1;
      else
        Migliori.num = 3;
*/
      while ((NumRighe > 0) && (NumColonne > 0))
      { for (l = 1; l <= Migliori.num; l++)
        { Migliori.col[l] = 0;
          Migliori.Val[l] = -DBL_MAX; };

        /* DISATTIVATA!
        if (NumColonne <= MINI_COL_CORR)
          Peggiori.num = 1;
        else
          Peggiori.num = NumColonne/DIVI_COL_CORR;
        for (l = 1; l <= Peggiori.num; l++)
        { Peggiori.col[l] = 0;
          Peggiori.Val[l] = DBL_MAX; };
        */

        /* PASSO 7 : Fissa le colonne di valore primale e duale molto promettente. */
        for (j = Ac[0].succ; j > 0; j = tempcol)
        { tempcol = Ac[j].succ;
          if (DS[j] >= 0.99 && CRP[j] <= 0.01)
            ColonnaOttima(j);
          else
          { Valore = DS[j]-ALFA*CRP[j];
            // AggiornaPeggiori(&Peggiori,Valore,j); DISATTIVATA!
            AggiornaMigliori(&Migliori,Valore,j);
          };
        };

        /* PASSO 8: Fissa la colonna migliore o una delle tre migliori.
                    Elimina un certo numero delle colonne peggiori. */
/*        if (Iterazioni == 1)
          ColonnaOttima(Migliori.col[1]);
        else // ATTENZIONE: E se vi fossero meno di tre colonne?
        { prob = ran1(&dummy);
          if (prob < PR1)
            ColonnaOttima(Migliori.col[3]);
          else if (prob < PR1 + PR2)
            ColonnaOttima(Migliori.col[2]);
          else
            ColonnaOttima(Migliori.col[1]);
        };
*/
        prob = ran1(&dummy);
        l = ceil(prob * Migliori.num);
        ColonnaOttima(Migliori.col[l]);
        // ATTENZIONE: E se vi fossero meno di Migliori.num colonne?
        // Bisognerebbe moltiplicare prob per (NumColonne) e sommare al risultato di
        // 'ceil' il valore Migliori.num - NumColonne.

        /*********** PASSO 9 : cancellazione colonne Peggiori - DISATTIVATA *************/
        /*
        for (k = 1; k <= Peggiori.num; k++)
        { j = Peggiori.col[k];
          // Controlla che la colonna non sia essenziale, benche' pessima.
          for (l = 1; (l <= Ac[j].num) && (Ar[Ac[j].elem[l]].num > 1); l++);
          if (l <= Ac[j].num)
            ColonnaOttima(j);
          else
            EliminaColonna(j);
        };
        */

        /* PASSO 13 : Se il problema e' piccolo, calcola le Penalita' Duali. */
        if ((NumRighe > 0) && (NumColonne <= COL_PEN_DUALI))
          DualAscentPenalties(UZMax,ZUB);

        /* PASSO 10 : se il problema e' vuoto, vai al passo 16 */
        if (NumRighe > 0)
        { do /* PASSO 11+12 : Riduzioni interne */
          { RowDomination();
            j = ColumnInclusion();
            j += ColumnDomination();
#ifdef UNIF
            // RIPROVARE
            j += ColDomination();
#endif
          } while (j >= MinRid2);
        };

        if (NumRighe > 0)
        { /* PASSO 14 : Sottogradiente secondario */
          SubgradientAscent(PS,DS,CRP,CRD,&LUB,&LLB,IterSG2);
          printf("ZUB = %ld, UB = %ld, LB = %ld\n",(long) ZUB, (long) LUB, (long) LLB);
          if (ZUB <= LLB) break;
        }
      }

      if ((NumRighe == 0) && (OUB < ZUB))
      { ZUB = 0.0;
        Sol[1] = 0;
        CompletaSoluzione(Sol,&ZUB); }

#ifndef ZDD
      /* PASSO 16+17: Salva la copertura ottenuta. */
      lg = fopen(LOGFILE,"a");
      fprintf(lg,"Soluzione:\n");
      for (i = 1; Sol[i] > 0; i++)
        fprintf(lg,"%6d",Sol[i]);
      fprintf(lg,"\nCosto = %ld\n",(long) ZUB);
      fclose(lg);
#endif

      RestoreProblem();
    }
  }

#ifdef DOS
  fine = clock();
  //tempo = (double) (fine - inizio)/CLK_TCK;
  tempo = (double) (fine - inizio)/CLOCKS_PER_SEC;

#else
  fine = util_cpu_time();
  //tempo = (double) (fine - inizio)/CLK_TCK;
  tempo = (double) (fine - inizio)/CLOCKS_PER_SEC;

#endif

#ifdef ZDD
  cover = sm_row_alloc();
  if (cover)
  { for (i = 1; Sol[i] > 0; i++)
      sm_row_insert(cover, Sol[i]-1); }
  else
  { printf("Error allocating cover");
    exit(0); }
#else
  SalvaRisultati(Risultati);
#endif

  Dealloca();

#ifdef ZDD
  return cover;
#else
  return 0;
#endif
}

/********************** Fine del programma principale ***********************/


/**************************** LeggiIstruzioni *******************************/
#ifndef ZDD
static void LeggiIstruzioni(int argc,char *argv[])
{ char ris;

  switch(argc)
  { case 1: printf("Heuristic for Set Covering\n"
                   "Premi X per uscire, H per avere aiuto,"
                   " un altro tasto per procedere.\n");
             ris = toupper(getc(stdin));
             if (ris == 'X') exit(0);
             if (ris == 'H')
               if (!system("view scwolf.nfo"))
                 if (!system("type scwolf.nfo | more"))
                   printf("\nUnable to display the help file\a\n");
             printf("\nFile dei dati: ");
             //gets(Dati);
	     fgets(Dati, sizeof(Dati), stdin);
	     Dati[strcspn(Dati, "\n")] = '\0';
             printf("\nFile dei risultati: ");
             //gets(Risultati);
             fgets(Risultati, sizeof(Risultati), stdin);
	     Risultati[strcspn(Risultati, "\n")] = '\0';
             printf("\nMassimo numero di iterazioni: ");
             scanf("%d",&MaxIter);
             printf("\nGli altri parametri assumeranno i valori di default.\n");
            break;
    case 3: strcpy(Dati,argv[1]);
            strcpy(Risultati,argv[2]);
            break;
    case 4: strcpy(Dati,argv[1]);
            strcpy(Risultati,argv[2]);
            strcpy(Parametri,argv[3]);
            CaricaParametri(Parametri);
            break;
    default:printf("Usage: scg [Data] [Results] [Parameters]\n");
            exit(0);
  };
}
#endif

/******************************* CaricaDati **********************************/
#ifndef ZDD
static void CaricaDati(char *NomeFile)
/* Il formato dei dati e' il seguente:
   RigheIniz
   ColonneIniz
   3 righe di dati inutili
   Costi delle colonne
   Righe coperte dalla colonna 1
   Righe coperte dalla colonna 2
   ....
   Righe coperte dalla colonna ColonneIniz
*/
{ FILE *f;
  int i, j, l;
  int* Temp;
#ifdef DOS
  time_t giorno;
#endif

  f = fopen(NomeFile,"r");
  if (f == NULL)
  { printf("\nError opening %s.\n\a",NomeFile);
    exit(0); };

#ifdef DOS
  lg = fopen(LOGFILE,"a+");
  time(&giorno);
  fprintf(lg,"\nLog started at %s\n",ctime(&giorno));
  fprintf(lg,"\nNome del file di ingresso: %s.\n",NomeFile);
  fclose(lg);
#endif

  fscanf(f,"%d\n",&RigheIniz);
  fscanf(f,"%d\n",&ColonneIniz);

  /* Salta tre righe che contengono dati inutili. */
  fscanf(f,"%d\n",&l);
  fscanf(f,"%d\n",&l);
  fscanf(f,"%d\n",&l);

  Ar = (struct MatRC *) calloc(RigheIniz+1,sizeof(*Ar));
  if (Ar == NULL)
  { printf("Insufficient memory for Ar!");
    exit(0); }
  Ac = (struct MatCR *) calloc(ColonneIniz+1,sizeof(*Ac));
  if (Ac == NULL)
  { printf("Insufficient memory for Ac!");
    exit(0); }

  /* Legge i costi delle colonne. */
  for (j = 1; j <= ColonneIniz; j++)
    fscanf(f,"%d",&Ac[j].peso);
  fscanf(f,"\n");

  /* Alloca una variabile dove copieremo le righe coperte da ogni colonna
    in attesa di sapere quante sono, per poi copiarle in Ac[i]. Meglio
    sarebbe che i dati di ogni colonna cominciassero con il numero di righe
    coperte. */
  Temp = (int *) calloc(RigheIniz+2,sizeof(int));
  if (Temp == NULL)
  { printf("Insufficient memory for Temp!");
    exit(0); }

  /* Legge le righe coperte da ciascuna colonna. */
  for (j = 1; j <= ColonneIniz; j++)
  { l = 0;
    do { fscanf(f,"%d",&Temp[++l]); }
    while (Temp[l] > 0);
    fscanf(f,"\n");
    Ac[j].num = l-1;
    Ac[j].iniziali = l-1;
    Ac[j].last = l;

    /* Alloca Num+2 elementi perche' c'e' anche uno 0 finale. */
    Ac[j].elem = (int *) calloc(Ac[j].num+2,sizeof(*Ac[j].elem));
    if (Ac[j].elem == NULL)
    { printf("Insufficient memory for Ac[%d]!",j);
      exit(0); }

    for (l = 1; l <= Ac[j].num; l++)
    { Ac[j].elem[l] = Temp[l];
      Ar[Temp[l]].num++; };
    Ac[j].elem[l] = 0;

    Ac[j].succ = j+1;
    Ac[j].prec = j-1;
  };
  Ac[0].succ = 1;
  Ac[ColonneIniz].succ = 0;

  if (Temp != NULL) free(Temp);

  /* Aggiorna le colonne che coprono ciascuna riga. */
  for (i = 1; i <= RigheIniz; i++)
  { /* Alloca Num+2 elementi perche' c'e' anche uno 0 finale. */
    Ar[i].set = (int *) calloc(Ar[i].num+2,sizeof(*Ar[i].set));
    if (Ar[i].set == NULL)
    { printf("Insufficient memory for Ar[%d]!",i);
      exit(0); }
    Ar[i].iniziali = Ar[i].num;
    Ar[i].last = 1;
  };

  for (j = 1; j <= ColonneIniz; j++)
  { for (l = 1; l <= Ac[j].num; l++)
    { /* La colonna j copre la riga i = Ac[j].elem[l],
         ed e' la Ar[i].last-esima a farlo. */
      i = Ac[j].elem[l];
      Ar[i].set[Ar[i].last] = j;
      Ar[i].last++;
    };
  };

  for (i = 1; i <= RigheIniz; i++)
  { Ar[i].set[Ar[i].last] = 0;
    Ar[i].succ = i+1;
    Ar[i].prec = i-1;
  };
  Ar[0].succ = 1;
  Ar[RigheIniz].succ = 0;

  fclose(f);
};
#endif

/**************************** CaricaParametri *******************************/
#ifndef ZDD
static void CaricaParametri(char *NomeFile)
{ FILE *f;

  f = fopen(NomeFile,"r");
  if (f == NULL)
  { printf("\nError opening %s.\n\a",NomeFile);
    exit(0); };

  lg = fopen(LOGFILE,"a+");
  fprintf(lg,"\nNome del file di parametri: %s.\n",NomeFile);
  fclose(lg);

  if ( (fscanf(f,"%*[^=]=%d",&IterSG1) < 1) ||
       (fscanf(f,"%*[^=]=%d",&IterSG2) < 1) ||
       (fscanf(f,"%*[^=]=%d",&MinRid) < 1) ||
       (fscanf(f,"%*[^=]=%d",&MinRid2) < 1) ||
       (fscanf(f,"%*[^=]=%d",&DIVI_COL_CORR) < 1) ||
       (fscanf(f,"%*[^=]=%d",&MINI_COL_CORR) < 1) ||
       (fscanf(f,"%*[^=]=%lf",&F0) < 1) ||
       (fscanf(f,"%*[^=]=%d",&ALFA) < 1) ||
       (fscanf(f,"%*[^=]=%lf",&PR1) < 1) ||
       (fscanf(f,"%*[^=]=%lf",&PR2) < 1) ||
       (fscanf(f,"%*[^=]=%lf",&PR3) < 1) ||
       (fscanf(f,"%*[^=]=%lf",&MIN_GAP) < 1) ||
       (fscanf(f,"%*[^=]=%d",&MaxIter) < 1) ||
       (fscanf(f,"%*[^=]=%lf",&MaxTempo) < 1) )
  { printf("\nParameter format error.\n\a");
    exit(0); };

  fclose(f);
};
#endif

/********************************* Alloca ***********************************/

static void Alloca(void)
{ int j;

  PS = (double *) calloc(RigheIniz+1,sizeof(double));
  if (PS == NULL)
  { printf("Insufficient memory for PS!");
    exit(0); }

  CRD = (double *) calloc(RigheIniz+1,sizeof(double));
  if (CRD == NULL)
  { printf("Insufficient memory for CRD!");
    exit(0); }

  DS = (double *) calloc(ColonneIniz+1,sizeof(double));
  if (DS == NULL)
  { printf("Insufficient memory for DS!");
    exit(0); }

  CRP = (double *) calloc(ColonneIniz+1,sizeof(double));
  if (CRP == NULL)
  { printf("Insufficient memory for CRP!");
    exit(0); }

  Migliori.col = (int *) calloc(MaxIter+1,sizeof(int));
  if (Migliori.col == NULL)
  { printf("Insufficient memory for Migliori.col!");
    exit(0); }

  Migliori.Val = (double *) calloc(MaxIter+1,sizeof(double));
  if (Migliori.Val == NULL)
  { printf("Insufficient memory for Migliori.Val!");
    exit(0); }

  j = max(ColonneIniz/DIVI_COL_CORR,1);
  Peggiori.col = (int *) calloc(j+1,sizeof(int));
  if (Peggiori.col == NULL)
  { printf("Insufficient memory for Peggiori.col!");
    exit(0); }
  Peggiori.Val = (double *) calloc(j+1,sizeof(double));
  if (Peggiori.Val == NULL)
  { printf("Insufficient memory for Peggiori.Val!");
    exit(0); }

  LX = (int *) calloc(ColonneIniz+1,sizeof(int));
  if (LX == NULL)
  { printf("Insufficient memory for LX!");
    exit(0); }

  PG = (int *) calloc(RigheIniz+1,sizeof(int));
  if (PG == NULL)
  { printf("Insufficient memory for PG!");
    exit(0); }

  LU = (int *) calloc(RigheIniz+1,sizeof(int));
  if (LU == NULL)
  { printf("Insufficient memory for LU!");
    exit(0); }

  DG = (int *) calloc(ColonneIniz+1,sizeof(int));
  if (DG == NULL)
  { printf("Insufficient memory for DG!");
    exit(0); }

  /* ColumnDomination */
#ifdef DOS
  Usata = (enum bool *) calloc(ColonneIniz+1,sizeof(*Usata));
#else
  Usata = (int *) calloc(ColonneIniz+1,sizeof(*Usata));
#endif
  if (Usata == NULL)
  { printf("Insufficient memory for Usata!");
    exit(0); }

  /* Soluzioni: possono essere ridondanti, per cui il massimo e' NumColonne!
                Il +2 viene dal fatto che le soluzioni terminano con uno 0. */
  Sol = (int *) calloc(ColonneIniz+2,sizeof(int));
  if (Sol == NULL)
  { printf("Insufficient memory for Sol!");
    exit(0); }

  SolLUB = (int *) calloc(ColonneIniz+2,sizeof(int));
  if (SolLUB == NULL)
  { printf("Insufficient memory for SolLUB!");
    exit(0); }

  Ridondanti = (int *) calloc(ColonneIniz+2,sizeof(int));
  if (Ridondanti == NULL)
  { printf("Insufficient memory for Ridondanti!");
    exit(0); }

  /* UpperBound */
  cover = (int *) calloc(RigheIniz+1,sizeof(int));
  if (cover == NULL)
  { printf("Insufficient memory for cover!");
    exit(0); }

  covermio = (int *) calloc(RigheIniz+1,sizeof(int));
  if (cover == NULL)
  { printf("Insufficient memory for covermio!");
    exit(0); }


  Agg = (int *) calloc(ColonneIniz+1,sizeof(int));
  if (Agg == NULL)
  { printf("Insufficient memory for Agg!");
    exit(0); }

  DaAgg = (int *) calloc(ColonneIniz+1,sizeof(int));
  if (DaAgg == NULL)
  { printf("Insufficient memory for DaAgg!");
    exit(0); }

  data = (struct DatiCol *) calloc(ColonneIniz+1,sizeof(*data));
  if (data == NULL)
  { printf("Insufficient memory for data!");
    exit(0); }

  /* DualAscent */
  N = (int *) calloc(RigheIniz+1,sizeof(int));
  if (N == NULL)
  { printf("Insufficient memory for N!");
    exit(0); }

  /* SubgradientAscent */
  UZMax = (double *) calloc(RigheIniz+1,sizeof(double));
  if (UZMax == NULL)
  { printf("Insufficient memory for UZMax!");
    exit(0); }

  /* DualAscentPenalities */
  UR = (double *) calloc(RigheIniz+1,sizeof(double));
  if (UR == NULL)
  { printf("Insufficient memory for UR!");
    exit(0); }
}


/****************************** Inizializza *********************************/

static void Inizializza(void)
{ int i, j, l, MaxCop;

  NumRighe = RigheIniz;
  NumColonne = ColonneIniz;

  Ac[0].peso = INT_MAX;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  { Ar[i].ColMin = 0;
    for (l = 1; l <= Ar[i].num; l++)
    { j = Ar[i].set[l];
      // ATTENZIONE: Qui si potrebbe considerare, a parita' di peso, il numero
      //             massimo di righe coperte.
      if (Ac[j].peso < Ac[Ar[i].ColMin].peso) Ar[i].ColMin = j; };
  };

  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  { PS[i] = 0.0;
    CRD[i] = 1.0; }

  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
  { DS[j] = 0.0;
    CRP[j] = (double) Ac[j].peso; }

  Funzioni[1] = &Funz1;
  Funzioni[2] = &Funz2;
  Funzioni[3] = &Funz3;
  Funzioni[4] = &Funz4;

  /* UpperBound richiede il logaritmo del numero di righe coperte da ogni
     colonna. Per dimensionare, serve il massimo numero di righe coperte. */
  MaxCop = Ac[1].num;
  for (j = 2; j <= ColonneIniz; j++)
    if (Ac[j].num > MaxCop) MaxCop = Ac[j].num;

  Log2 = (double *) calloc(MaxCop+1,sizeof(double));
  if (Log2 == NULL)
  { printf("Insufficient memory for Log2!");
    exit(0); }
  cLog2 = (double *) calloc(MaxCop+1,sizeof(double));
  if (cLog2 == NULL)
  { printf("Insufficient memory for cLog2!");
    exit(0); }

  // RIPROVARE: Bisogna distinguere i casi l = 1 e l = 2! Eppure peggiora!
  Log2[1] = 1.0;
  cLog2[1] = 1.0;
  for (l = 2; l <= MaxCop; l++)
  { Log2[l] = log2((double) l);
    cLog2[l] = (double) l * Log2[l]; };
//  for (l = 1; l <= MaxCop; l++)
//  { Log2[l] = log2((double) l+1);
//    cLog2[l] = (double) l * Log2[l]; };
}


/******************************* RowDomination *******************************/

static int RowDomination(void)
{ int i1, i2, j, jMin, l;
#ifdef DOS
  enum bool Continua;
#else
  int Continua;
#endif
  int OldRighe = NumRighe;

  Ac[0].num = INT_MAX;
  for (i1 = Ar[0].succ; i1 > 0; i1 = temprighe)
  { temprighe = Ar[i1].succ;
    // Limita la ricerca alle righe della piu' corta colonna coprente i1.
    jMin = 0;
    for (l = 1; l <= Ar[i1].num; l++)
    { j = Ar[i1].set[l];
      if (Ac[j].num < Ac[jMin].num) jMin = j; };

    Continua = VERO;
    for (l = 1; (l <= Ac[jMin].num) && (Continua); l++)
    { i2 = Ac[jMin].elem[l];
      /* ATTENZIONE: Se RigaDomina desse la riga dominante e potessi eliminarla
                 senza problemi di 'temprighe', potrei testare solo i1 < i2. */
      if (RigaDomina(i1,i2))
      { EliminaRiga(i2);
//        Continua = FALSO;
        l--;
        RowDom++; };
    };
  };

  return OldRighe-NumRighe;
};


/******************************* RigaDomina *********************************/
#ifdef DOS
static enum bool RigaDomina(int i1,int i2)
#else
static int RigaDomina(int i1,int i2)
#endif
{ int l1, l2;

  /* Se una riga e' piu' lunga, non puo' dominarne un'altra.
     Una riga domina se stessa, ma cio' non autorizza ad eliminarla!  */
  if ((Ar[i1].num > Ar[i2].num) || (i1 == i2)) return FALSO;

  /* Cerca gli elementi della riga i1 nella riga i2. Se li trova tutti,
     i1 domina i2; altrimenti, no. */
  for (l1 = 1; l1 <= Ar[i1].num; l1++)
  { for (l2 = 1; (l2 <= Ar[i2].num) && (Ar[i1].set[l1] != Ar[i2].set[l2]); l2++);
    if (l2 > Ar[i2].num) return FALSO; };
  return VERO;
};


/**************************** ColumnInclusion *******************************/

static int ColumnInclusion(void)
{ int OldIncl;
  int i;

  OldIncl = ColIncl;
  for (i = Ar[0].succ; i > 0; i = temprighe)
  { temprighe = Ar[i].succ;
    if (Ar[i].num == 1)
    { ColonnaOttima(Ar[i].set[1]);
      ColIncl++; };
  };
  return ColIncl-OldIncl;
};


/**************************** ColumnDomination ******************************/

static int ColumnDomination(void)
{ int j, j2, l, OldDom;
  int UB;

  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
    Usata[j] = FALSO;

  /* Per ogni colonna cerca una copertura migliore con le colonne di ColMin. */
  OldDom = ColDom;
  for (j = Ac[0].succ; j > 0; j = tempcol)
  { tempcol = Ac[j].succ;

    UB = 0;
    for (l = 1; (l <= Ac[j].num) && (UB <= Ac[j].peso); l++)
    { j2 = Ar[Ac[j].elem[l]].ColMin;
      if (!Usata[j2])
      { UB += Ac[j2].peso;
        Usata[j2] = VERO; }
      if (j2 == j) UB = INT_MAX;
    };

    // Ripulisce Usata. Va fatto prima di eliminare la colonna!
    for (l = 1; l <= Ac[j].num; l++)
    { j2 = Ar[Ac[j].elem[l]].ColMin;
      Usata[j2] = FALSO; }

    if (UB <= Ac[j].peso)
    { EliminaColonna(j);
      ColDom++; };
  };

  return ColDom-OldDom;
};


static int ColDomination(void)
{ int j1, j2, i, iMin, l;
#ifdef DOS
  enum bool Continua;
#else
  int Continua;
#endif
  int OldCol = NumColonne;

  Ar[0].num = INT_MAX;
  for (j1 = Ac[0].succ; j1 > 0; j1 = tempcol)
  { tempcol = Ac[j1].succ;
    // Limita la ricerca alle colonne della piu' corta riga coperta da j1.
    iMin = 0;
    for (l = 1; l <= Ac[j1].num; l++)
    { i = Ac[j1].elem[l];
      if (Ar[i].num < Ar[iMin].num) iMin = i; };

    Continua = VERO;
    for (l = 1; (l <= Ar[iMin].num) && (Continua); l++)
    { j2 = Ar[iMin].set[l];
      if (ColonnaDomina(j2,j1))
      { EliminaColonna(j1);
        Continua = FALSO;
        ColDom++; };
    };
  };

  return OldCol - NumColonne;
};

#ifdef DOS
static enum bool ColonnaDomina(int j1, int j2)
#else
static int ColonnaDomina(int j1, int j2)
#endif
{ int l1, l2;

  /* Se una colonna e' piu' corta o costa di piu', non puo' dominarne un'altra.
     Una colonna domina se stessa, ma cio' non autorizza ad eliminarla!  */
  if ((Ac[j1].num < Ac[j2].num) || (Ac[j1].peso > Ac[j2].peso) || (j1 == j2)) return FALSO;

  /* Cerca gli elementi della colonna j2 nella colonna j1. Se li trova tutti,
     j1 domina j2; altrimenti, no. */
  for (l2 = 1; l2 <= Ac[j2].num; l2++)
  { for (l1 = 1; (l1 <= Ac[j1].num) && (Ac[j1].elem[l1] != Ac[j2].elem[l2]); l1++);
    if (l1 > Ac[j1].num) return FALSO; };
  return VERO;
};


/****************************** EliminaColonna *******************************/

static void EliminaColonna(int j)
{ int l, k, i, j2;

  /* Se la colonna e' essenziale non viene eliminata, ma, anzi, fissata. */
  for (l = 1; l <= Ac[j].num; l++)
    if (Ar[Ac[j].elem[l]].num == 1)
    { ColonnaOttima(j);
      return; };

  /* Conserva il puntatore, in caso di eliminazioni multiple. */
  if (j == tempcol) tempcol = Ac[j].succ;

  NumColonne--;
  if (Ac[j].succ > 0) Ac[Ac[j].succ].prec = Ac[j].prec;
  Ac[Ac[j].prec].succ = Ac[j].succ;

  for (l = 1; l <= Ac[j].num; l++)
  { i = Ac[j].elem[l];
    Ar[i].num--;

    /* Cancella la colonna dall'elenco Ar[i].set. */
    for (k = 1; Ar[i].set[k] != j; k++);
    Ar[i].set[Ar[i].last] = Ar[i].set[k];
    Ar[i].last--;
    Ar[i].set[k] = Ar[i].set[Ar[i].last];
    Ar[i].set[Ar[i].last] = 0;

    if (Ar[i].num == 0) EliminaRiga(i);

    /* Aggiorna ColMin, se la colonna eliminata era la colonna minima coprente. */
    if (Ar[i].ColMin == j)
    { Ac[0].peso = INT_MAX;
      Ar[i].ColMin = 0;
      for (k = 1; k <= Ar[i].num; k++)
      { j2 = Ar[i].set[k];
        // RIPROVARE: a tener conto anche del numero di righe coperte.
        if (Ac[j2].peso < Ac[Ar[i].ColMin].peso)
        /*if ( (Ac[j2].peso < Ac[Ar[i].ColMin].peso) ||
            ((Ac[j2].peso == Ac[Ar[i].ColMin].peso) &&
             (Ac[j2].num > Ac[Ar[i].ColMin].peso)))*/
          Ar[i].ColMin = j2;
      };
    };
  };
  EliCol++;
};


/******************************* EliminaRiga ********************************/

static void EliminaRiga(int i)
{ int l,k,j;

  /* Conserva il puntatore, in caso di eliminazioni multiple. */
  if (i == temprighe) temprighe = Ar[i].succ;

  if (Ar[i].succ != 0) Ar[Ar[i].succ].prec = Ar[i].prec;
  Ar[Ar[i].prec].succ = Ar[i].succ;

  NumRighe--;
  for (l = 1; Ar[i].set[l] != 0; l++)
  { j = Ar[i].set[l];
    Ac[j].num--;

    /* Cancella la riga dall'elenco Ac[j].elem. */
    for (k = 1; Ac[j].elem[k] != i;  k++);
    Ac[j].elem[Ac[j].last] = Ac[j].elem[k];
    Ac[j].last--;
    Ac[j].elem[k] = Ac[j].elem[Ac[j].last];
    Ac[j].elem[Ac[j].last] = 0;

    if (Ac[j].num == 0) EliminaColonna(j);
  };
  EliRow++;
};


/***************************** ColonnaOttima ********************************/

static void ColonnaOttima(int j)
{ int i, k, l;

  /* Conserva il puntatore, in caso di eliminazioni multiple. */
  // ATTENZIONE: Per qualche motivo, ero convinto che questo fosse inutile.
  //             Non ricordo piu'.
  if (j == tempcol) tempcol = Ac[j].succ;

  NumColonne--;

  if (Ac[j].succ != 0) Ac[Ac[j].succ].prec = Ac[j].prec;
  Ac[Ac[j].prec].succ = Ac[j].succ;


  for (l = 1; Ac[j].elem[l] != 0; l++)
  { i = Ac[j].elem[l];
    Ar[i].num--;

    /* Cancella la colonna dall'elenco Ar[i].set. */
    for (k = 1; Ar[i].set[k] != j; k++);
    Ar[i].set[Ar[i].last] = Ar[i].set[k];
    Ar[i].last--;
    Ar[i].set[k] = Ar[i].set[Ar[i].last];
    Ar[i].set[Ar[i].last] = 0;

    /* E' fondamentale cancellare la colonna dalle righe prima di eliminarle.
       Altrimenti, eliminare le righe puo' far scattare EliminaColonna su j. */
    EliminaRiga(i);
  };

  Ac[j].prec = 0;
  Ac[j].succ = colott;
  colott = j;
  OUB += (double) Ac[j].peso;

  OptCol++;
};


/***************************** CopiaSoluzione *******************************/

static void CopiaSoluzione(int* Sol1,double Val1,int* Sol2,double* Val2)
{ int l;
  *Val2 = Val1;
  for (l = 1; Sol1[l] != 0; l++)
    Sol2[l] = Sol1[l];
  Sol2[l] = 0;
};


/**************************** CompletaSoluzione *****************************/

static void CompletaSoluzione(int* Sol, double* Val)
{ int l, j;
  for (l = 1; Sol[l] != 0; l++);
  for (j = colott; j != 0; j = Ac[j].succ)
    Sol[l++] = j;
  *Val += OUB;
  Sol[l] = 0;
};


/******************************* UpperBound *********************************/
/*                                                                          */
/* Euristica di Set Covering di E. Balas e A. Ho: ad ogni passo ci sceglie  */
/* la colonna per la quale e' minima una data funzione del peso (c) e del   */
/* numero di nuove righe coperte (n). Sono definite tre funzioni:           */
/* 1) c / n                                                                 */
/* 2) c / lg(n)                                                             */
/* 3) c / n lg(n)                                                           */
/*                                                                          */
/****************************************************************************/

static double Funz1(double Cost, int Num)
{ return Cost/Num; };


static double Funz2(double Cost, int Num)
{ return Cost/Log2[Num]; };


static double Funz3(double Cost, int Num)
{ return Cost/cLog2[Num]; };


static double Funz4(double Cost, int j)
{ double sum;
  int i, j2, l, rcp, l2, tmp;

  CRP[0] = INT_MAX;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  if (cover[i] == 0)
  { Ar[i].ColMin = 0;
    for (l = 1; l <= Ar[i].num; l++)
    { j2 = Ar[i].set[l];
      // ATTENZIONE: Qui si potrebbe considerare, a parita' di peso, il numero
      //             massimo di righe coperte.
      if (CRP[j2] < CRP[Ar[i].ColMin]) Ar[i].ColMin = j2; };
  };

/* per ogni riga cerca la colonna migliore che la copre utilizzando CRP */


  for (j2 = data[0].succ; j2 > 0; j2 = data[j2].succ)
     Usata[j2] = FALSO;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    if (cover[i] == 0)
       covermio[i] = 0;
       
/* inizializza usata e covermio. covermio e' un vettore per segnare le righe coperte piu' volte */
       

  /* Per ogni colonna cerca una copertura migliore con le colonne di ColMin. */
 
  sum = 0.0;
  rcp = 0;
  for (l = 1; l <= Ac[j].num; l++)
  if (cover[Ac[j].elem[l]] == 0)
  { j2 = Ar[Ac[j].elem[l]].ColMin;
    if (!Usata[j2])
    { sum += CRP[j2];
      tmp = data[j2].num;    
      for (l2 = 1; l2 <= data[j2].num; l2++)
         if ((cover[Ac[j2].elem[l2]] == 0) && (covermio[Ac[j2].elem[l2]] > 0))
           tmp--;
/* toglie dal numero di righe coperte da ogni colonna le righe gia' coperte da colonne gia in sum */           
      rcp += tmp;  
      Usata[j2] = VERO;
      for (l2 = 1; (l2 <= data[j2].num); l2++)
        if (cover[Ac[j2].elem[l2]] == 0)
            covermio[Ac[j2].elem[l2]]++;
/* aggiorna le righe coperte in covermio */           
    }        
      //      if (j2 == j) sum = INT_MAX;
  };
  Cost=Cost/data[j].num;
  sum=sum/rcp;

/* rimette a posto Ar[i].ColMin che ha sporcato. Attenzione che questa operazione puo' essere fatta anche fuori */  
   
  Ac[0].peso = INT_MAX;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  { Ar[i].ColMin = 0;
    for (l = 1; l <= Ar[i].num; l++)
    { j = Ar[i].set[l];
      // ATTENZIONE: Qui si potrebbe considerare, a parita' di peso, il numero
      //             massimo di righe coperte.
      if (Ac[j].peso < Ac[Ar[i].ColMin].peso) Ar[i].ColMin = j; };
  };
  
  return 2*Cost - sum;
};


static void EliminaRidondanti(int *Solution)
{ int i, j;

  j = 1;
  for (i = 1; Solution[i] != 0; i++)
    if (Solution[i] >  0)
    { Solution[j] = Solution[i];
      j++; };
  Solution[j] = 0;
};


static void UpperBound(double *Cost, int Funz, int *Solution, double* UB)
// Assume che Solution contenga tutte e sole le colonne con Cost <= 0. La
// completa con alcune colonne aventi Cost > 0 e poi la rende non
// ridondante eliminando le colonne piu' costose rispetto al vettore Ac.cost.
{ int i, j, j2, k, l;
  int jOtt, kOtt;
  int Sol_num;                 /* numero di colonne nella soluzione corrente */
  int Scoperte = 0;

  *UB = DBL_MAX;
  Solution[1] = 0;

  /* Inizializzazione dei dati che descrivono il sottoproblema:
     se si parte da una soluzione lagrangiana, Solution e' una copertura
     parziale, e Cost sono i costi lagrangiani relativi. Bisogna eliminare le
     colonne di Solution e segnare le righe gia' coperte. */
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    cover[i] = 0;

  Sol_num = 0;
  data[0].ratio = DBL_MAX;
  data[0].prec = Ac[0].prec;
  j2 = 0;
  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
  { if (Cost[j] > 0.0)
    { data[j].num = Ac[j].num;
      data[j].prec = j2;
      data[j2].succ = j;
      j2 = j;
    }
    else
    { Solution[++Sol_num] = j;
      for (l = 1; l <= Ac[j].num; l++)
        cover[Ac[j].elem[l]]++;
    }
  };
  data[j2].succ = 0;

  Scoperte = NumRighe;
  if (Sol_num > 0)
  { for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
      if (cover[i] > 0)
      { Scoperte--;
        for (l = 1; l <= Ar[i].num; l++)
          data[Ar[i].set[l]].num--;
      };

    // Questo dovrebbe essere assorbito dalle riduzioni.
    for (j = data[0].succ; j > 0; j = data[j].succ)
      if (data[j].num <= 0)
      { data[data[j].succ].prec = data[j].prec;
        data[data[j].prec].succ = data[j].succ; }
  };

  /* Calcolo dei costi */
  for (j = data[0].succ; j > 0; j = data[j].succ)
    if (Funz == 4)
      data[j].ratio = Funzioni[Funz](Cost[j],j);
    else
      data[j].ratio = Funzioni[Funz](Cost[j],data[j].num);


  while ((Scoperte > 0) && (Sol_num < NumColonne))
  { /* Scelta della colonna migliore */
    jOtt = 0;
    for (j = data[0].succ; j > 0; j = data[j].succ)
      if (data[j].ratio < data[jOtt].ratio) jOtt = j;

    if (jOtt == 0) return;

    Solution[++Sol_num] = jOtt;

    /* Aggiornamento delle righe */
    NumAgg = 0;
    for(l = 1; l <= Ac[jOtt].num; l++)
    { i = Ac[jOtt].elem[l];
      cover[i]++;
      if (cover[i] == 1)
      { Scoperte--;
        for (k = 1; k <= Ar[i].num; k++)
        { Agg[Ar[i].set[k]]++;
          // ATTENZIONE: Non basta ridurre data[Ar[i].set[k]].num di 1?
          if (Agg[Ar[i].set[k]] == 1) DaAgg[++NumAgg] = Ar[i].set[k]; };
      };
    };

    /* Aggiornamento delle colonne */
    for (k = 1; k <= NumAgg; k++)
    { j = DaAgg[k];
      data[j].num -= Agg[j];
      if (data[j].num > 0)
      { if (Funz == 4)
          data[j].ratio = Funzioni[Funz](Cost[j],j);
        else
          data[j].ratio = Funzioni[Funz](Cost[j],data[j].num);
      }
      else
      { data[data[j].succ].prec = data[j].prec;
        data[data[j].prec].succ = data[j].succ; }
      Agg[j] = 0;
    }
  }
  Solution[Sol_num+1] = 0;

  if (Scoperte > 0) return;


  NumRidond = 0;
  for (k = 1; k <= Sol_num; k++)
  { j = Solution[k];
    if (j > 0)
    { for (l = 1; (l <= Ac[j].num) && (cover[Ac[j].elem[l]] > 1); l++);
      if (l > Ac[j].num) Ridondanti[++NumRidond] = k; };
  };
  Ridondanti[NumRidond+1] = 0;

  /* Eliminazione delle colonne ridondanti */
  while (NumRidond > 0)
  { /* Elimina la colonna ridondante piu' costosa. */
    Ac[0].peso = -1;
    kOtt = 0;
    for (l = 1; l <= NumRidond; l++)
      if (Ac[Solution[Ridondanti[l]]].peso > Ac[Solution[kOtt]].peso)
        kOtt = Ridondanti[l];
    if (kOtt == 0) break;

    j = Solution[kOtt];
    /* Aggiornamento di 'cover' */
    for (l = 1; l <= Ac[j].num; l++)
      cover[Ac[j].elem[l]]--;

    Solution[kOtt] = -1;
    Sol_num--;

    NumRidond = 0;
    for (k = 1; Ridondanti[k] != 0; k++)
    { j = Solution[Ridondanti[k]];
      if (j > 0)
      { for (l = 1; (l <= Ac[j].num) && (cover[Ac[j].elem[l]] > 1); l++);
        if (l > Ac[j].num) Ridondanti[++NumRidond] = Ridondanti[k]; };
    };
    Ridondanti[NumRidond+1] = 0;
  };

  if (kOtt > 0) EliminaRidondanti(Solution);

  *UB = 0.0;
  for (l = 1; l <= Sol_num; l++)
    *UB += Ac[Solution[l]].peso;
};


static void UpperBoundRid(double *Cost, int Funz, int *Solution, double* UB)
// Da' per scontato che Solution derivi dall'accettare le colonne con Cost <= 0.
// Completa Solution aggiungendovi alcune colonne con Cost > 0 e poi la rende
// non ridondante eliminando le colonne piu' costose rispetto al vettore Ac.cost.
// Fa anche le riduzioni via via che procede: opera sulla matrice di copertura,
// per cui la salva nelle apposite strutture dati (quindi si puo' usare solo
// fuori del ciclo principale).
{ int i, j, k, l;
  int jOtt, kOtt;
  int Sol_num;                /* numero di colonne nella soluzione corrente */

  SaveProblem();

  *UB = DBL_MAX;

  /* Inizializzazione dei dati che descrivono il sottoproblema */
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    cover[i] = 0;

  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
  { if (Cost[j] <= 0.0)
      // Se si parte da una soluzione lagrangiana con costi Cost, bisogna eliminare
      // le colonne di costo negativo e segnare le righe gia' coperte.
      // Si sistema cover prima di fissare le colonne perche' altrimenti alla
      // seconda colonna, la riga e' gia' eliminata!
      for (l = 1; l <= Ac[j].num; l++)
        cover[Ac[j].elem[l]]++;
  };

  Sol_num = 0;
  Solution[1] = 0;
  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
    if (Cost[j] <= 0.0)
    { Solution[++Sol_num] = j;
      ColonnaOttima(Solution[k]); };

  if (colott > 0)
  { do
    { RowDomination();
      j = ColumnInclusion();
      j += ColumnDomination();
      j += ColDomination();
    } while (j >= MinRid2);
  };

  /* Calcolo dei costi */
  // ATTENZIONE: PORCATA! (usare data, intendo)
  data[0].ratio = DBL_MAX;
  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
    if (Funz == 4)
      data[j].ratio = Funzioni[Funz](Cost[j],j);
    else
      data[j].ratio = Funzioni[Funz](Cost[j],Ac[j].num);

  while ((NumRighe > 0) && (NumColonne > 0))
  { /* Scelta della colonna migliore */
    jOtt = 0;
    for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
      if (data[j].ratio < data[jOtt].ratio) jOtt = j;

    if (jOtt == 0) return;

    ColonnaOttima(jOtt);

    do
    { RowDomination();
      j = ColumnInclusion();
      j = ColumnDomination();
      j += ColDomination();
    } while (j >= MinRid2);

    for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
      if (Funz == 4)
        data[j].ratio = Funzioni[Funz](Cost[j],j);
      else
        data[j].ratio = Funzioni[Funz](Cost[j],Ac[j].num);
  };

  if (NumRighe > 0) return;


  Sol_num = 0;
  for (j = colott; j > 0; j = Ac[j].succ)
    Solution[++Sol_num] = j;
  Solution[Sol_num+1] = 0;

  RestoreProblem();

  /* Aggiornamento delle righe */
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    cover[i] = 0;

  for (l = 1; l <= Sol_num; l++)
  { j = Solution[l];
    for (k = 1; k <= Ac[j].num; k++)
    { i = Ac[j].elem[k];
      cover[i]++; };
  };

  // Determinazione delle colonne ridondanti
  NumRidond = 0;
  for (k = 1; k <= Sol_num; k++)
  { j = Solution[k];
    if (j > 0)
    { for (l = 1; (l <= Ac[j].num) && (cover[Ac[j].elem[l]] > 1); l++);
      if (l > Ac[j].num) Ridondanti[++NumRidond] = k; };
  };
  Ridondanti[NumRidond+1] = 0;

  /* Eliminazione delle colonne ridondanti */
  while (NumRidond > 0)
  { /* Elimina la colonna ridondante piu' costosa. */
    Ac[0].peso = -1;
    kOtt = 0;
    for (l = 1; l <= NumRidond; l++)
      if (Ac[Solution[Ridondanti[l]]].peso > Ac[Solution[kOtt]].peso) kOtt = Ridondanti[l];
    if (kOtt == 0) break;

    j = Solution[kOtt];
    /* Aggiornamento di 'cover' */
    for (l = 1; l <= Ac[j].num; l++)
      cover[Ac[j].elem[l]]--;

    Solution[kOtt] = -1;
    Sol_num--;

    NumRidond = 0;
    for (k = 1; Ridondanti[k] != 0; k++)
    { j = Solution[Ridondanti[k]];
      if (j > 0)
      { for (l = 1; (l <= Ac[j].num) && (cover[Ac[j].elem[l]] > 1); l++);
        if (l > Ac[j].num) Ridondanti[++NumRidond] = Ridondanti[k]; };
    };
    Ridondanti[NumRidond+1] = 0;
  };

  if (kOtt > 0) EliminaRidondanti(Solution);

  *UB = 0.0;
  for (l = 1; l <= Sol_num; l++)
    *UB += Ac[Solution[l]].peso;
};


/******************************* DualAscent *********************************/

static void DualAscent(double *U, double *LB)
{ int i, l;
  double temp, minda;

  DualAsc++;

  /* N contiene ordina le righe per numero di colonne coprenti crescente. */
  l = 0;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    N[++l] = i;

  qsort((void *)&N[1],NumRighe,sizeof(N[0]),NumeroColonne);

  /* PASSO 1: considero le righe in ordine decrescente. */
  for (l = NumRighe; l > 0; l--)
  { i = N[l];

    minda = minDA(U,i);
    temp = U[i] + min(minda,0);
    U[i] = max(temp,0);
  };

  /* PASSO 2: considero le righe in ordine crescente. */
  for (l = 1; l <= NumRighe; l++)
  { i = N[l];
    U[i] += minDA(U,i); };

  /* calcolo del lower bound */
  *LB = OUB;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    *LB += U[i];
};


static double minDA(double *U, int i)
{ double min, temp;
  int j, k, l;

  min = DBL_MAX;
  for (l = 1; l <= Ar[i].num; l++)
  { j = Ar[i].set[l];
    temp = (double) Ac[j].peso;
    for (k = 1; k <= Ac[j].num; k++)
      temp -= U[Ac[j].elem[k]];
    if (temp < min) min = temp;
  };
  return min;
};


static int NumeroColonne(const void *A, const void *B)
{ int a, b;

  a = (*(int *) A);
  b = (*(int *) B);
  if (Ar[a].num > Ar[b].num) return 1;
  if (Ar[a].num < Ar[b].num) return -1;
  // Se le colonne sono ordinate per costi crescenti e le righe in modo che le
  // prime siano coperte dalle prime colonne, dovrebbe esser meglio usare le
  // prime righe per saturare i vincoli duali piu' stretti.
  if (a < b) return 1;
  if (a > b) return -1;
  return 0;
};


/**************************** SubgradientAscent *****************************/

static void SubgradientAscent(double *PS, double *DS, double *CRP, double *CRD,
                              double *LUB, double *LLB, int IterSG)
{ int i, j, l, lp = 0;
  double LB, UB;             /* bound derivati dai rilassamenti lagrangiani */
  double ZLBmax = -DBL_MAX, ZUBmin = DBL_MAX;        /* migliori bound noti */
  double UBeur;        /* bound derivanti dal completamento del lagrangiano */
  double Norma;
  double pf = 2.0, df = 2.0;
  double pT, dT;                     /* passo di incremento primale e duale */
  int pK = 0, dK = 0;


  *LUB = DBL_MAX;

// ATTENZIONE: La riga seguente serve nei problemi a costi non interi:
//while ((pf >= F0 || df >= F0) && (ZUBmin-ZLBmax >= MIN_GAP))
  while ((pf >= F0 || df >= F0) && (ceil(ZLBmax) < ceil(ZUBmin)))
  { SubGrad++;
    //RIPROVARE: la limitazione del passo
    if (pf >= F0) SolvePLLB(PS,LX,CRP,&LB);
    if (df >= F0) SolveDLUB(DS,LU,CRD,&UB);
    //SolvePLLB(PS,LX,CRP,&LB);
    //SolveDLUB(DS,LU,CRD,&UB);

    lp = LagrangeanPenalties(CRP,LB,ZUB);
    // RIPROVARE: pare non cambi nulla.
    //lp = LagrangeanRowSplitting(CRP,LB,ZUB);

    //RIPROVARE:
    if (pf >= F0) {
    //RIPROVARE:
    //if ((ZLBmax < LB) || (lp < 0))
    if ((ceil(ZLBmax) < ceil(LB)) || (lp < 0))
    { ZLBmax = LB;
      for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
        UZMax[i] = PS[i];
      pK = 1; }
    else
     pK++;
    };

    //RIPROVARE:
    if (df >= F0) {
    if (ZUBmin <= UB)
      dK++;
    else
    { ZUBmin = UB;
      dK = 1; }
    };

    //RIPROVARE:
    if (pf >= F0) {
    if (pK > IterSG)
    { pK = 1;
      pf /= 2.0; };
    };

    //RIPROVARE:
    if (df >= F0) {
    if (dK > IterSG)
    { dK = 1;
      df /= 2.0; };
    };

    //RIPROVARE:
    if (pf >= F0) {
    /* Euristica lagrangiana di completamento (trova un nuovo upper bound). */
    UpperBound(/*NULL,*/CRP,rotate,SolLUB,&UBeur);
//    rotate++;
//    rotate = ((rotate == 5) ? 1 : rotate);
//    rotate = ((rotate == 1) ? 2 : 1);

    /* Aggiorna l'upper bound lagrangiano e il migliore upper bound intero. */
    if (UBeur + OUB < *LUB) *LUB = UBeur + OUB;
    if (UBeur + OUB < ZUB)
    { CopiaSoluzione(SolLUB,UBeur,Sol,&ZUB);
      CompletaSoluzione(Sol,&ZUB); }

    // RIPROVARE: Prova a sfruttare tutte le euristiche.
    /*for (rotate = 1; rotate <= 3; rotate++)
    { UpperBound(CRP,rotate,SolLUB,&UBeur);
      // Aggiorna l'upper bound lagrangiano e il migliore upper bound intero.
      if (UBeur + OUB < *LUB) *LUB = UBeur + OUB;
      if (UBeur + OUB < ZUB)
      { CopiaSoluzione(SolLUB,UBeur,Sol,&ZUB);
        CompletaSoluzione(Sol,&ZUB); };
    };*/
    };

    /* Se upper bound = lower bound, abbiamo trovato l'ottimo. */
    if (ceil(ZLBmax)== ZUB) break;

    //RIPROVARE: Adesso pare migliori leggermente.
/*    if (ZUB < ZUBmin)
    { ZUBmin = ZUB;
      for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
        DS[j] = 0;
      for (l = 1; Sol[l] > 0; l++)
        DS[Sol[l]] = 1;
      //df = 2.0;
    };
*/
    //RIPROVARE:
    if (pf >= F0) {
    /* PROBLEMA PRIMALE:                                                    */
    /* 1) Calcolo dei sottogradienti primali                                */
    Norma = 0.0;
    for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    { PG[i] = 1;
      for (l = 1; l <= Ar[i].num; l++)
        PG[i] -= LX[Ar[i].set[l]];

      if ((PS[i] == 0.0) && (PG[i] < 0))
        PG[i] = 0;
      else
        Norma += (double) PG[i]*PG[i];
    };

    /* 2) Calcolo del passo primale                                         */
    if (Norma == 0.0)
      pT = 0.0;
    else
      pT = pf * (ZUBmin - LB) / Norma;

    /* 3) Aggiornamento dei moltiplicatori primali.                         */
    for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
      PS[i] = max(PS[i] + pT * PG[i] , 0);
    };

    //RIPROVARE:
    if (df >= F0) {
    /* PROBLEMA DUALE:                                                      */
    /* 1) Calcolo dei sottogradienti duali                                  */
    Norma = 0.0;
    for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
    { DG[j] = -Ac[j].peso;
      for (l = 1; l <= Ac[j].num; l++)
        DG[j] += LU[Ac[j].elem[l]];

      if ((DS[j] == 0.0) && (DG[j] < 0))
        DG[j] = 0;
      else
        Norma += (double) DG[j]*DG[j];
    };

    /* 2) Calcolo del passo duale                                           */
    if (Norma == 0.0)
      dT = 0.0;
    else
      dT = df * (UB - ZLBmax) / Norma;

    /* 3) Aggiornamento dei moltiplicatori duali.                           */
    for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
      DS[j] = max(DS[j] + dT * DG[j] , 0);
    };

    if (lp != 0)
    { do {
        RowDomination();
        j = ColumnInclusion();
        j += ColumnDomination();
#ifdef UNIF
        // RIPROVARE
        j += ColDomination();
#endif
      } while (j >= MinRid2);
    };
  };

  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    PS[i] = UZMax[i];
  *LLB = ceil(ZLBmax);
};


/********************** SolvePrimalLagrangeanLowerBound **********************/

static void SolvePLLB(double *PS, int *LX, double *CRP, double *LB)
{ int i, j, l;

  *LB = OUB;
  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
  { CRP[j] = (double) Ac[j].peso;
    for (l = 1; l <= Ac[j].num; l++)
      CRP[j] -= PS[Ac[j].elem[l]];

    if (CRP[j] > 0.0)
      LX[j] = 0;
    else
    { LX[j] = 1;
      *LB += CRP[j]; };
  };

  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
    *LB += PS[i];
};


/*********************** SolveDualLagrangeanUpperBound ***********************/

static void SolveDLUB(double *DS, int *LU, double *CRD, double *UB)
{ int i, j, l;

  *UB = OUB;
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  { CRD[i] = 1.0;
    for (l = 1; l <= Ar[i].num; l++)
      CRD[i] -= DS[Ar[i].set[l]];

    if (CRD[i] <= 0.0)
      LU[i] = 0;
    else
    { LU[i] = Ac[Ar[i].ColMin].peso;
      *UB += CRD[i] * LU[i]; }
  };

  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
    *UB += DS[j] * Ac[j].peso;
};


/************************** LagrangeanPenalties *****************************/

static int LagrangeanPenalties(double *LC, double LB, double UB)
{ int j;
  int OldLagrDel = LagrPenDel;
  int OldLagrOpt = LagrPenOpt;

  for (j = Ac[0].succ; j > 0; j = tempcol)
  { tempcol = Ac[j].succ;
    if (LC[j] >= 0.0)
    { if (LB + LC[j] >= UB)
      { EliminaColonna(j);
        LagrPenDel++; };
    }
    else
    { if (LB - LC[j] >= UB)
      { ColonnaOttima(j);
        LagrPenOpt++; };
    };
  };
  if (LagrPenOpt > OldLagrOpt) return OldLagrOpt - LagrPenOpt;
  return LagrPenDel-OldLagrDel;
};


/************************ LagrangeanRowSplitting ****************************/

static int ConfrontaLC(const void *A, const void *B)
{ int a, b;

  a = (*(int *) A);
  b = (*(int *) B);
  if (CRP[a] > CRP[b]) return 1;
  if (CRP[a] < CRP[b]) return -1;
  if (a > b) return 1;
  if (a < b) return -1;
  return 0;
};


static void EliminaElemento(int i, int j)
{ int k;

  EliElem++;

  Ac[j].num--;
  /* Cancella la riga dall'elenco Ac[j].elem. */
  for (k = 1; Ac[j].elem[k] != i;  k++);
  Ac[j].elem[Ac[j].last] = Ac[j].elem[k];
  Ac[j].last--;
  Ac[j].elem[k] = Ac[j].elem[Ac[j].last];
  Ac[j].elem[Ac[j].last] = 0;
  if (Ac[j].num == 0) EliminaColonna(j);


  Ar[i].num--;
  /* Cancella la colonna dall'elenco Ar[i].set. */
  for (k = 1; Ar[i].set[k] != j; k++);
  Ar[i].set[Ar[i].last] = Ar[i].set[k];
  Ar[i].last--;
  Ar[i].set[k] = Ar[i].set[Ar[i].last];
  Ar[i].set[Ar[i].last] = 0;
  if (Ar[i].num == 0) EliminaRiga(i);
};


static int LagrangeanRowSplitting(double *LC, double LB, double UB)
{ int i, l, j, k, ll;
  int LagrRowSplit = 0;
  double somma;

  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  { somma = 0.0;
    k = 0;
    for (l = 1; l <= Ar[i].num; l++)
    { j = Ar[i].set[l];
      if (LC[j] <= 0) // pero' col costo nullo non aiuta: buttarle
      { somma -= LC[j];
        ColNeg[++k] = j; }
    };

    if (somma >= UB - LB)
    { qsort((void *)&ColNeg[1],k,sizeof(int),ConfrontaLC);
      somma = 0.0;
      for (l = 1; somma < UB - LB; l++)
        somma -= LC[ColNeg[l]];

      for (ll = l+1; ll < k; ll++)
        EliminaElemento(i,ColNeg[ll]);
    }
  };

  return LagrRowSplit;
};


/************************** DualAscentPenalities ****************************/

static int DualAscentPenalties(double *U,double UB)
{ int i, j, l;
  int peso;
  double LB;
  int OldColonne = NumColonne;

  for (j = Ac[0].succ; j > 0; j = tempcol)
  { tempcol = Ac[j].succ;

    /* OUT-PENALTY */
    peso = Ac[j].peso;
    for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
      UR[i] = U[i];

    Ac[j].peso = INT_MAX;
    DualAscent(UR,&LB);
    // Bisogna ristabilire il peso prima di ColonnaOttima per non sporcare OUB.
    Ac[j].peso = peso;

    if (LB >= UB)
    { ColonnaOttima(j);
      DAPenOpt++; }
    else
    { /* IN-PENALTY */
      for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
        UR[i] = U[i];

      // Azzerare il peso azzera anche le variabili duali, ma solo via via
      // che si incontrano, non fin dal principio!
      for (l = 1; l <= Ac[j].num; l++)
        UR[Ac[j].elem[l]] = 0.0;
      peso = Ac[j].peso;
      Ac[j].peso = 0;
      DualAscent(UR,&LB);
      Ac[j].peso = peso;

      if (LB + Ac[j].peso >= UB)
      { EliminaColonna(j);
        DAPenDel++; };
    };
  };
  return OldColonne - NumColonne;
};


/**************************** AggiornaPeggiori ******************************/

static void AggiornaPeggiori(struct Blocco *Peggiori, double Val, int j)
// Aggiunge j al Blocco delle colonne Peggiori, conservando l'ordine, in base
// al valore Val.
{ int l;

  // Retrocede di un passo tutte le colonne battute da j.
  for (l = 1; (l <= Peggiori->num) && (Peggiori->Val[l] > Val); l++)
  { Peggiori->Val[l-1] = Peggiori->Val[l];
    Peggiori->col[l-1] = Peggiori->col[l]; }
  Peggiori->col[l-1] = j;
  Peggiori->Val[l-1] = Val;
};


/**************************** AggiornaMigliori ******************************/

static void AggiornaMigliori(struct Blocco *Migliori, double Val, int j)
// Aggiunge j al Blocco delle colonne Migliori, conservando l'ordine, in base
// al valore Val.
{ int l;

  // Retrocede di un passo tutte le colonne battute da j.
  for (l = 1; (l <= Migliori->num) && (Migliori->Val[l] < Val); l++)
  { Migliori->Val[l-1] = Migliori->Val[l];
    Migliori->col[l-1] = Migliori->col[l]; }
  Migliori->col[l-1] = j;
  Migliori->Val[l-1] = Val;
};


/******************************* SalvaRisultati *********************************/

static void SalvaRisultati(char *NomeFile)
{
#ifndef ZDD
  FILE *f;
  int i, j;
//  int l;

  f = fopen(NomeFile,"w+");
  if (f == NULL)
  { printf("\nError opening %s.\n\a",NomeFile);
    return; };

  fprintf(f,"\nSOLUZIONE:\n");
  fprintf(f,"Costo = %ld\n",(long) ZUB);
  fprintf(f,"Tempo di esecuzione = %.2lf\n",tempo);
  for (i = 1; Sol[i] > 0; i++);
  fprintf(f,"Number of columns = %d\n",i-1);

  for (i = 1; Sol[i] > 0; i++)
  { j = Sol[i];
    fprintf(f,"%6d %6d",i,j);
// Tolgo la stampa delle righe coperte per ottenere una stampa canonica!
//    for (l = 1; l <= Ac[j].num; l++)
//      fprintf(f,"%6d ",Ac[j].elem[l]);
//    for (l = Ac[j].num+2; l <= Ac[j].iniziali+1; l++)
//      fprintf(f,"%6d ",Ac[j].elem[l]);
    fprintf(f,"%6d\n",0);
  }
  fprintf(f,"\n"
            "################################################\n"
            "# Numero di colonne originario =        %6d #\n"
            "# Numero di colonne finale =            %6d #\n"
            "# Numero di righe originario =          %6d #\n"
            "# Numero di righe finale =              %6d #\n"
            "################################################\n"
            "# Numero di colonne eliminate =         %6d #\n"
            "# Numero di righe eliminate =           %6d #\n"
            "# Numero di elementi eliminati =        %6d #\n"
            "# Numero di colonne essenziali =        %6d #\n"
            "# TEST DI RIDUZIONE:                           #\n"
            "# Column Domination =                   %6d #\n"
            "# Lagrangean Penalties =                %6d #\n"
            "# Dual Ascent Penalties =               %6d #\n"
            "# Column Inclusion =                    %6d #\n"
            "# Row Domination =                      %6d #\n"
            "# Iterazioni di Sottogradiente =        %6ld #\n"
            "# Iterazioni di Dual Ascent =           %6ld #\n"
            "################################################\n",
      ColonneIniz,NumColonne,RigheIniz,NumRighe,EliCol,EliRow,EliElem,OptCol,
      ColDom,LagrPenOpt+LagrPenDel,DAPenOpt+DAPenDel,ColIncl,RowDom,
      SubGrad,DualAsc);
  fclose(f);
#endif
  printf("\n"
            "################################################\n"
            "# Numero di colonne originario =        %6d #\n"
            "# Numero di colonne finale =            %6d #\n"
            "# Numero di righe originario =          %6d #\n"
            "# Numero di righe finale =              %6d #\n"
            "################################################\n"
            "# Numero di colonne eliminate =         %6d #\n"
            "# Numero di righe eliminate =           %6d #\n"
            "# Numero di elementi eliminati =        %6d #\n"
            "# Numero di colonne essenziali =        %6d #\n"
            "# TEST DI RIDUZIONE:                           #\n"
            "# Column Domination =                   %6d #\n"
            "# Lagrangean Penalties =                %6d #\n"
            "# Dual Ascent Penalties =               %6d #\n"
            "# Column Inclusion =                    %6d #\n"
            "# Row Domination =                      %6d #\n"
            "# Iterazioni di Sottogradiente =        %6ld #\n"
            "# Iterazioni di Dual Ascent =           %6ld #\n"
            "################################################\n",
      ColonneIniz,NumColonne,RigheIniz,NumRighe,EliCol,EliRow,EliElem,OptCol,
      ColDom,LagrPenOpt+LagrPenDel,DAPenOpt+DAPenDel,ColIncl,RowDom,
      SubGrad,DualAsc);
  printf("Costo = %ld\n",(long) ZUB);
  printf("Tempo di esecuzione = %.2lf\n",tempo);
};


/******************************** Dealloca **********************************/

static void Dealloca(void)
{ int i, j;

  if (Ar != NULL)
  { for (i = 1; i <= RigheIniz; i++)
      if (Ar[i].set != NULL) free(Ar[i].set);
    free(Ar); };

  if (Ac != NULL)
  { for (j = 1; j <= ColonneIniz; j++)
      if (Ac[j].elem != NULL) free(Ac[j].elem);
    free(Ac); };

  if (Log2 != NULL) free(Log2);
  if (cLog2 != NULL) free(cLog2);

  if (PS != NULL) free(PS);
  if (CRD != NULL) free(CRD);
  if (DS != NULL) free(DS);
  if (CRP != NULL) free(CRP);
  if (Migliori.col != NULL) free(Migliori.col);
  if (Migliori.Val != NULL) free(Migliori.Val);
  if (Peggiori.col != NULL) free(Peggiori.col);
  if (Peggiori.Val != NULL) free(Peggiori.Val);
  if (LX != NULL) free(LX);
  if (PG != NULL) free(PG);
  if (LU != NULL) free(LU);
  if (DG != NULL) free(DG);
  if (Usata != NULL) free(Usata);
  if (Sol != NULL) free(Sol);
  if (SolLUB != NULL) free(SolLUB);
  if (Ridondanti != NULL) free(Ridondanti);
  if (cover != NULL) free(cover);
  if (Agg != NULL) free(Agg);
  if (DaAgg != NULL) free(DaAgg);
  if (data != NULL) free(data);
  if (N != NULL) free(N);
  if (UZMax != NULL) free(UZMax);
  if (UR != NULL) free(UR);

  if (SAc != NULL) free(SAc);
  if (SAr != NULL) free(SAr);
  if (SPS != NULL) free(SPS);
  if (SDS != NULL) free(SDS);
  if (SCRP != NULL) free(SCRP);
  if (SCRD != NULL) free(SCRD);
};


/******************************* SaveProblem ********************************/

static void SaveProblem(void)
/* Conserva il problema corrente. Viene chiamata solo dopo il sottogradiente
   primario, per cui conserva solo il problema di partenza, con le riduzioni
   valide globalmente. RestoreProblem, quindi, torna sempre ad esso.        */
{ int i, j, l;

  SAr = (struct MatRC *) calloc(RigheIniz+1,sizeof(*SAr));
  if (SAr == NULL)
  { printf("Insufficient memory for SAr!");
    exit(0); }
  SAc = (struct MatCR *) calloc(ColonneIniz+1,sizeof(*SAc));
  if (SAc == NULL)
  { printf("Insufficient memory for SAc!");
    exit(0); }

  for (j = 0; j <= ColonneIniz; j++)
  { SAc[j] = Ac[j];

    SAc[j].elem = NULL;
    if (Ac[j].elem != NULL)
    { // Alloca Num+2 elementi perche' c'e' anche uno 0 finale.
      SAc[j].elem = (int *) calloc(Ac[j].iniziali+2,sizeof(int));
      if (SAc[j].elem == NULL)
      { printf("Insufficient memory for SAc[%d]!",j);
        exit(0); }

      for (l = 1; l <= Ac[j].iniziali + 1; l++)
        SAc[j].elem[l] = Ac[j].elem[l];
    };
  };

  for (i = 0; i <= RigheIniz; i++)
  { SAr[i] = Ar[i];

    SAr[i].set = NULL;
    if (Ar[i].set != NULL)
    { // Alloca Num+2 elementi perche' c'e' anche uno 0 finale.
      SAr[i].set = (int *) calloc(Ar[i].iniziali+2,sizeof(*Ar[i].set));
      if (SAr[i].set == NULL)
      { printf("Insufficient memory for SAr[%d]!",i);
        exit(0); }

      for (l = 1; l <= Ar[i].iniziali + 1; l++)
        SAr[i].set[l] = Ar[i].set[l];
    };
  };

  Srighecorr = NumRighe;
  Scolonnecorr = NumColonne;
  Scolott = colott;
  SOUB = OUB;

  SPS = (double *) calloc(RigheIniz+2,sizeof(double));
  if (SPS == NULL)
  { printf("Insufficient memory for SPS!");
    exit(0); }
  SDS = (double *) calloc(ColonneIniz+2,sizeof(double));
  if (SDS == NULL)
  { printf("Insufficient memory for SDS!");
    exit(0); }

  SCRP = (double *) calloc(ColonneIniz+2,sizeof(double));
  if (SCRP == NULL)
  { printf("Insufficient memory for SCRP!");
    exit(0); }
  SCRD = (double *) calloc(RigheIniz+2,sizeof(double));
  if (SCRD == NULL)
  { printf("Insufficient memory for SCRD!");
    exit(0); }

  for (j = Ac[0].succ; j > 0; j = Ac[j].succ)
  { SDS[j] = DS[j];
    SCRP[j] = CRP[j]; };
  for (i = Ar[0].succ; i > 0; i = Ar[i].succ)
  { SPS[i] = PS[i];
    SCRD[i] = CRD[i]; };

  SEliCol = EliCol;
  SEliRow = EliRow;
  SEliElem = EliElem;
  SOptCol = OptCol;
  SColDom = ColDom;
  SLagrPenOpt = LagrPenOpt;
  SLagrPenDel = LagrPenDel;
  SDAPenOpt = DAPenOpt;
  SDAPenDel = DAPenDel;
  SColIncl = ColIncl;
  SRowDom = RowDom;
  SSubGrad = SubGrad;
  SDualAsc = DualAsc;
};


/****************************** RestoreProblem *******************************/

static void RestoreProblem(void)
{ int i, j, l;

  NumRighe = Srighecorr;
  NumColonne = Scolonnecorr;
  colott = Scolott;
  OUB = SOUB;

  for (j = 0; j <= ColonneIniz; j++)
  { Ac[j].peso = SAc[j].peso;
    Ac[j].last = SAc[j].last;
    Ac[j].iniziali = SAc[j].iniziali;
    Ac[j].num = SAc[j].num;
    Ac[j].prec = SAc[j].prec;
    Ac[j].succ = SAc[j].succ;
    if (SAc[j].iniziali > 0)
      for (l = 1; l <= SAc[j].iniziali + 1; l++)
        Ac[j].elem[l] = SAc[j].elem[l];
  };

  for (i = 0; i <= RigheIniz; i++)
  { Ar[i].last = SAr[i].last;
    Ar[i].iniziali = SAr[i].iniziali;
    Ar[i].num = SAr[i].num;
    Ar[i].ColMin = SAr[i].ColMin;
    Ar[i].prec = SAr[i].prec;
    Ar[i].succ = SAr[i].succ;
    if (SAr[i].iniziali > 0)
      for (l = 1; l <= Ar[i].iniziali + 1; l++)
        Ar[i].set[l] = SAr[i].set[l];
  };

  for (j = SAc[0].succ; j > 0; j = SAc[j].succ)
  { DS[j] = SDS[j];
    CRP[j] = SCRP[j]; };
  for (i = SAr[0].succ; i > 0; i = SAr[i].succ)
  { PS[i] = SPS[i];
    CRD[i] = SCRD[i]; };

  EliCol = SEliCol;
  EliRow = SEliRow;
  EliElem = SEliElem;
  OptCol = SOptCol;
  ColDom = SColDom;
  LagrPenOpt = SLagrPenOpt;
  LagrPenDel = SLagrPenDel;
  DAPenOpt = SDAPenOpt;
  DAPenDel = SDAPenDel;
  ColIncl = SColIncl;
  RowDom = SRowDom;
  SubGrad = SSubGrad;
  DualAsc = SDualAsc;
};


/******************************************************************************
*                       GENERATORE DI NUMERI CASUALI                          *
******************************************************************************/

static float ran1(long *idum)
{ int j;
  long k;
  static long iy = 0;
  static long iv[NTAB];

  if (*idum <= 0 || !iy)
  { if (-(*idum) < 1)
      *idum = 1;
    else
      *idum = -(*idum);
    for (j = NTAB+7; j >= 0; j--)
    { k = (*idum)/IQ;
      *idum = IA*(*idum-k*IQ) - IR*k;
      if (*idum < 0) *idum += IM;
      if (j < NTAB) iv[j] = *idum;
    }
    iy = iv[0];
  }
  k = (*idum)/IQ;
  *idum = IA*(*idum-k*IQ) - IR*k;
  if (*idum <  0) *idum += IM;
  j = iy/NDIV;
  iy = iv[j];
  iv[j] = *idum;
  if (AM*iy > RNMX)
    return RNMX;
  else
    return AM*iy;
}

#ifdef DEBUG

#endif
