**********************************************************************************************
**                                 USER SUBROUTINE USDFLD                                   **
**********************************************************************************************
      SUBROUTINE USDFLD(FIELD,STATEV,PNEWDT,DIRECT,T,CELENT,
     1 TIME,DTIME,CMNAME,ORNAME,NFIELD,NSTATV,NOEL,NPT,LAYER,
     2 KSPT,KSTEP,KINC,NDI,NSHR,COORD,JMAC,JMATYP,MATLAYO,LACCFLA)

      INCLUDE 'ABA_PARAM.INC'
 
      CHARACTER*80 CMNAME,ORNAME
      CHARACTER*3  FLGRAY(15)
      DIMENSION FIELD(NFIELD),STATEV(NSTATV),DIRECT(3,3),
     1 T(3,3),TIME(2)
      DIMENSION ARRAY(15),JARRAY(15),JMAC(*),JMATYP(*),COORD(*)	  
      
      INCLUDE 'InputOrientFortran2.f'
      CF = FVF0(NPT,NOEL)
C      y = COORD(2)
	  
      FIELD(1) = CF
      STATEV(1) = 217870*CF+ 1550*(1-CF)
      RETURN
      END


      SUBROUTINE ORIENT(T,NOEL,NPT,LAYER,KSPT,COORDS,BASIS,
     1                  ORNAME,NNODES,CNODES,JNNUM)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 ORNAME
      CHARACTER*256 JOBNAME, OUTDIR, FILENAME
C
      DIMENSION T(3,3),COORDS(3),BASIS(3,3),CNODES(3,NNODES)
      DIMENSION JNNUM(NNODES)
                 
      INCLUDE 'InputOrientFortran1.f'
            
      T(1,1) =  DCOS(PHI0(NPT,NOEL))
      T(1,2) = -DSIN(PHI0(NPT,NOEL))
      T(1,3) = 0.
      T(2,1) =  DSIN(PHI0(NPT,NOEL))
      T(2,2) =  DCOS(PHI0(NPT,NOEL))
      T(2,3) = 0.
      T(3,1) = 0.
      T(3,2) = 0.
      T(3,3) = 1.

      RETURN
      END  