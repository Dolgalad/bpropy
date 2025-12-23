/*****************************************************************************
* Euristica per la risoluzione del Set Covering - Definizioni                *
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <limits.h>

//#define DEBUG // Induce la compilazione di alcune procedure ausiliarie.
#define DOS
//#define ZDD
//#define UNIF

#ifdef ZDD
#include "sparse.h"
#endif

#ifdef DOS
#include <float.h>
enum bool {FALSO = 0, VERO = 1};
#else
#include "util.h"
#define FALSO 0
#define VERO 1
#endif

#ifndef ZDD
#define LOGFILE "scpino.log"
#endif

/**************** PARAMETRI DEL GENERATORE DI NUMERI CASUALI ****************/

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


/*************************** DEFINIZIONI DI TIPO ****************************/

struct MatRC { int *set;
               int last;
               int iniziali;
               int num;
               int ColMin;   /* Colonna di costo minimo che copre ogni riga */
               int prec;
               int succ;
             };

struct MatCR { int *elem;
               int peso;
               int last;
               int iniziali;
               int num;
               int prec;
               int succ;
             };

struct Blocco { int num;
                int *col;
                double *Val; };

struct DatiCol { double ratio;
                 int num;
                 int succ;
                 int prec; };

#ifdef ZDD
extern struct MatRC *Ar;               /* colonne che coprono ciascuna riga */
extern struct MatCR *Ac;               /* righe coperte da ciascuna colonna */

sm_row* SCG(int RigheIniz, int ColonneIniz);
#endif

/****************************************************************************/
