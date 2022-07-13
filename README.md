---
author: David Johnson
title: STG external interpreter
date: 07/12/2022
---

### The External STG Interpreter
  - A new Haskell STG interpreter
  - [Csaba's talk](https://www.youtube.com/watch?v=Ey5OFPkxF_w)

---

### GHC Pipeline

![](https://user-images.githubusercontent.com/875324/138091797-b573c198-85ef-4a57-9d14-0b96df826ff0.png)

---

### STG

  - Graph reduction language
  - Strict, higher-order, lambda calculus, closure-converted, but not lambda-lifted.

---

### STG AST
![](https://user-images.githubusercontent.com/875324/178803355-9556af3e-d04d-49fd-97f7-74637d3d2bd2.png)

---

### STG Syntax

```haskell
-- | A top-level binding.
data TopBinding' idBnd idOcc dcOcc tcOcc
-- See Note [CoreSyn top-level string literals]
  = StgTopLifted    (Binding' idBnd idOcc dcOcc tcOcc)
  | StgTopStringLit idBnd BS.ByteString
  deriving (Eq, Ord, Generic, Show)

data Binding' idBnd idOcc dcOcc tcOcc
  = StgNonRec idBnd (Rhs' idBnd idOcc dcOcc tcOcc)
  | StgRec    [(idBnd, Rhs' idBnd idOcc dcOcc tcOcc)]
  deriving (Eq, Ord, Generic, Show)

data Arg' idOcc
  = StgVarArg  idOcc
  | StgLitArg  !Lit
  deriving (Eq, Ord, Generic, Show)

data Rhs' idBnd idOcc dcOcc tcOcc
  = StgRhsClosure
		[idOcc]                   -- non-global free vars
		!UpdateFlag               -- ReEntrant | Updatable | SingleEntry
		[idBnd]                   -- arguments; if empty, then not a function;
								  -- as above, order is important.
		(Expr' idBnd idOcc dcOcc tcOcc) -- body

  | StgRhsCon
		dcOcc  -- DataCon
		[Arg' idOcc]        -- Args
  deriving (Eq, Ord, Generic, Show)

data Expr' idBnd idOcc dcOcc tcOcc
  = StgApp
		idOcc         -- function
		[Arg' idOcc]  -- arguments; may be empty
		Type          -- result type
		(Name,Name,Name)  -- fun core type pp, result core type pp, StgApp oigin (Var/Coercion/App)
  | StgLit      Lit
		-- StgConApp is vital for returning unboxed tuples or sums
		-- which can't be let-bound first
  | StgConApp dcOcc      -- DataCon
				[Arg' idOcc]  -- Saturated
				[Type]        -- types
  | StgOpApp    StgOp         -- Primitive op or foreign call
				[Arg' idOcc]  -- Saturated.
				Type          -- result type
				(Maybe tcOcc) -- result type name (required for tagToEnum wrapper generator)
  | StgCase
		(Expr' idBnd idOcc dcOcc tcOcc)     -- the thing to examine
		idBnd                               -- binds the result of evaluating the scrutinee
		(AltType' tcOcc)
		[Alt' idBnd idOcc dcOcc tcOcc]      -- The DEFAULT case is always *first*
  | StgLet
		(Binding' idBnd idOcc dcOcc tcOcc)  -- right hand sides (see below)
		(Expr' idBnd idOcc dcOcc tcOcc)     -- body
  | StgLetNoEscape
		(Binding' idBnd idOcc dcOcc tcOcc)  -- right hand sides (see below)
		(Expr' idBnd idOcc dcOcc tcOcc)     -- body
  | StgTick
		Tickish
		(Expr' idBnd idOcc dcOcc tcOcc)     -- sub expression
  deriving (Eq, Ord, Generic, Show)
```

---

### Let no escape

The STG program has a new construct called let-no-escape, that encodes so-called join points. Variables bound by a let-no-escape are guaranteed to be tail-calls, not embedded inside a data structure, in which case we don’t have to construct a closure because the required stack will always be present. Todo: say more.

---

### STG PrimOp Calls

```haskell
data StgOp
  = StgPrimOp     !Name
  | StgPrimCallOp !PrimCall
  | StgFCallOp    !ForeignCall
  deriving (Eq, Ord, Generic, Show)
```

---

### GHC WPC pipeline
![](https://user-images.githubusercontent.com/875324/178640354-3b5e0697-5f4d-4230-a3f4-6f6675d6c9f2.png)

---

### Interpret GHC programs

  - Demo time

---

### External STG Interpreter features

  - Supports most GHC primops and RTS features.
	 - GC, Threads, I/O Manager
  - Debugger, call graph visualizer

---

### External STG Interpreter motivation

  - From scratch implementation of the STG machine in Haskell.
  - Supports most GHC primops and RTS features.
  - Tool to study the runtime behaviour of Haskell programs, i.e. it can run/interpret GHC or Pandoc.
  - The implementation of the interpreter is in plain simple Haskell, so it makes compiler backend and tooling development approachable for everyone.
  - Debugger which supports step-by-step evaluation, breakpoints and execution region based inspection.

---

### GHC-WPC

  - Run real world Haskell programs that were compiled with GHC Whole Program Compiler (GHC-WPC).
  - GHC-WPC is a GHC fork that exports the whole program STG IR.

---

### STG papers

  - [Canonical paper](https://www.microsoft.com/en-us/research/publication/implementing-lazy-functional-languages-on-stock-hardware-the-spineless-tagless-g-machine)
  - [Faster laziness through dynamic pointer tagging](https://www.microsoft.com/en-us/research/publication/faster-laziness-using-dynamic-pointer-tagging/)
  - [How to make a fast curry](https://www.microsoft.com/en-us/research/publication/make-fast-curry-pushenter-vs-evalapply/)

---

### Push / enter vs. eval / apply.

How to resolve unknown function calls. The tldr; is `push/enter` expects the callee to inpect stack arguments and closure saturation, `eval/apply` expects the calling function to do that. In practice GHC has found `eval/apply` to be better. See "How to make a fast curry".

```haskell
zipWith :: (a->b->c) -> [a] -> [b] -> [c]
zipWith k [] [] = []
zipWith k (x:xs) (y:ys) = kxy: zipWith k xs ys
```

Used to be `push/enter` is now `eval/apply`.

---

### Spineless and tagless

 - Spineless deals with no allocations during strict evaluation.
 - Tagless deals with /not/ tagging pointers with information about constructors.
   - GHC used to be tagless, but no longer is due to branch misprediction problems.

---

### STG interpreters

  - [MiniSTG](https://wiki.haskell.org/Ministg)
  - [stgi](https://hackage.haskell.org/package/stgi)
  - [external-stg-interpreter](https://github.com/grin-compiler/ghc-whole-program-compiler-project/tree/master/external-stg-interpreter)

---

### SPJ on STG

![Grainy SPJ movie on STG from over 10 years ago](https://www.youtube.com/watch?v=v0J1iZ7F7W8&list=PLBkRCigjPwyeCSD_DFxpd246YIF7_RDDI)

---

### ESTG tree -d

![](https://user-images.githubusercontent.com/875324/178643026-db60572b-1509-4201-9d8b-fa2db17684b3.png)

---

### ESTG

Exports entire Modules.

```haskell
data Module' idBnd idOcc dcOcc tcBnd tcOcc
  = Module
  { modulePhase               :: !BS8.ByteString
  , moduleUnitId              :: !UnitId
  , moduleName                :: !ModuleName
  , moduleSourceFilePath      :: !(Maybe Name) -- HINT: RealSrcSpan's source file refers to this value
  , moduleForeignStubs        :: !(ForeignStubs' idOcc)
  , moduleHasForeignExported  :: !Bool
  , moduleDependency          :: ![(UnitId, [ModuleName])]
  , moduleExternalTopIds      :: ![(UnitId, [(ModuleName, [idBnd])])]
  , moduleTyCons              :: ![(UnitId, [(ModuleName, [tcBnd])])]
  , moduleTopBindings         :: ![TopBinding' idBnd idOcc dcOcc tcOcc]
  , moduleForeignFiles        :: ![(ForeignSrcLang, FilePath)]
  }
  deriving (Eq, Ord, Generic, Show)
```

---

### ESTG Machine environment

  - Atoms and Heap Objects
  - StackContinuation
  - I/O Manager
  - FFI
  - Multithreaded Scheduler (single core)
  - Garbage collector
  - eval / apply evaluator.

```haskell
type Addr   = Int
type Heap   = IntMap HeapObject
type Env    = Map Id (StaticOrigin, Atom)   -- NOTE: must contain only the defined local variables
type Stack  = [StackContinuation]
```

---

### StackContinuation

```haskell
data StackContinuation
  = CaseOf  !Int !Id !Env !Binder !AltType ![Alt]
  -- ^ closure addr & name (debug) ; pattern match on the result ; carries the closure's local environment
  | Update  !Addr
  -- ^ update Addr with the result heap object ; NOTE: maybe this is irrelevant as the closure interpreter will perform the update if necessary
  | Apply   ![Atom]
  -- ^ apply args on the result heap object
  | Catch   !Atom !Bool !Bool
  -- ^ catch frame ; exception handler, block async exceptions, interruptible
  | RestoreExMask !Bool !Bool
  -- ^ saved: block async exceptions, interruptible
  | RunScheduler  !ScheduleReason
  | DataToTagOp
  | DebugFrame    !DebugFrame
  -- ^ for debug purposes, it does not required for STG evaluation
  deriving (Show, Eq, Ord)
```

---

### ESTG limitations

  - STM not implemented
  - GHCi ByteCode primop not implemented
  - No compact regions

---

### ESTG Machine

```haskell
type M a = StateT IO StgState a
```

### ESTG StgState

```haskell
data StgState
  = StgState
  { ssHeap                :: !Heap
  , ssStaticGlobalEnv     :: !Env   -- NOTE: top level bindings only!

  -- GC
  , ssLastGCAddr          :: !Int
  , ssGCInput             :: PrintableMVar ([Atom], StgState)
  , ssGCOutput            :: PrintableMVar RefSet
  , ssGCIsRunning         :: Bool

  -- let-no-escape support
  , ssTotalLNECount       :: !Int

  -- string constants ; models the program memory's static constant region
  -- HINT: the value is a PtrAtom that points to the key BS's content
  , ssCStringConstants    :: Map ByteString Atom

  -- threading
  , ssThreads             :: IntMap ThreadState

  -- thread scheduler related
  , ssCurrentThreadId     :: Int
  , ssScheduledThreadIds  :: [Int]  -- HINT: one round

  -- primop related

  , ssStableNameMap       :: Map Atom Int
  , ssWeakPointers        :: IntMap WeakPtrDescriptor
  , ssStablePointers      :: IntMap Atom
  , ssMutableByteArrays   :: IntMap ByteArrayDescriptor
  , ssMVars               :: IntMap MVarDescriptor
  , ssMutVars             :: IntMap Atom
  , ssArrays              :: IntMap (Vector Atom)
  , ssMutableArrays       :: IntMap (Vector Atom)
  , ssSmallArrays         :: IntMap (Vector Atom)
  , ssSmallMutableArrays  :: IntMap (Vector Atom)
  , ssArrayArrays         :: IntMap (Vector Atom)
  , ssMutableArrayArrays  :: IntMap (Vector Atom)

  , ssNextThreadId          :: !Int
  , ssNextHeapAddr          :: {-# UNPACK #-} !Int
  , ssNextStableName        :: !Int
  , ssNextWeakPointer       :: !Int
  , ssNextStablePointer     :: !Int
  , ssNextMutableByteArray  :: !Int
  , ssNextMVar              :: !Int
  , ssNextMutVar            :: !Int
  , ssNextArray             :: !Int
  , ssNextMutableArray      :: !Int
  , ssNextSmallArray        :: !Int
  , ssNextSmallMutableArray :: !Int
  , ssNextArrayArray        :: !Int
  , ssNextMutableArrayArray :: !Int

  -- FFI related
  , ssCBitsMap            :: DL
  , ssStateStore          :: PrintableMVar StgState

  -- RTS related
  , ssRtsSupport          :: Rts

  -- debug
  , ssIsQuiet             :: Bool
  , ssCurrentClosureEnv   :: Env
  , ssCurrentClosure      :: Maybe Id
  , ssCurrentClosureAddr  :: Int
  , ssExecutedClosures    :: !(Set Int)
  , ssExecutedClosureIds  :: !(Set Id)
  , ssExecutedPrimOps     :: !(Set Name)
  , ssExecutedFFI         :: !(Set ForeignCall)
  , ssExecutedPrimCalls   :: !(Set PrimCall)
  , ssHeapStartAddress    :: !Int
  , ssClosureCallCounter  :: !Int

  -- call graph
  , ssCallGraph           :: !CallGraph
  , ssCurrentProgramPoint :: !ProgramPoint

  -- debugger API
  , ssDebuggerChan        :: DebuggerChan
  , ssNextDebugCommand    :: NextDebugCommand

  , ssEvaluatedClosures   :: !(Set Name)
  , ssBreakpoints         :: !(Map Name Int)
  , ssDebugState          :: DebugState
  , ssStgErrorAction      :: Printable (M ())

  -- region tracker
  , ssMarkers             :: !(Map Name (Set Region))
  , ssRegions             :: !(Map Region (Maybe AddressState, CallGraph, [(AddressState, AddressState)]) )

  -- retainer db
  , ssReferenceMap        :: !(IntMap IntSet)
  , ssRetainerMap         :: !(IntMap IntSet)
  , ssGCRootSet           :: !IntSet

  -- tracing
  , ssTracingState        :: TracingState

  -- origin db
  , ssOrigin              :: !(IntMap (Id, Int, Int)) -- HINT: closure, closure address, thread id

  -- GC marker
  , ssGCMarkers           :: ![AddressState]

  -- tracing primops
  , ssTraceEvents         :: ![(String, AddressState)]
  , ssTraceMarkers        :: ![(String, AddressState)]
  }
  deriving (Show)
```

---

### ESTG Threading

```haskell
data AsyncExceptionMask
  = NotBlocked
  | Blocked     {isInterruptible :: !Bool}
  deriving (Eq, Ord, Show)

data ThreadState
  = ThreadState
  { tsCurrentResult     :: [Atom] -- Q: do we need this? A: yes, i.e. MVar read primops can write this after unblocking the thread
  , tsStack             :: ![StackContinuation]
  , tsStatus            :: !ThreadStatus
  , tsBlockedExceptions :: [Int] -- ids of the threads waitng to send an async exception
  , tsBlockExceptions   :: !Bool  -- block async exceptions
  , tsInterruptible     :: !Bool  -- interruptible blocking of async exception
--  , tsAsyncExMask     :: !AsyncExceptionMask
  , tsBound             :: !Bool
  , tsLocked            :: !Bool  -- Q: what is this for? is this necessary?
  , tsCapability        :: !Int   -- NOTE: the thread is running on this capability ; Q: is this necessary?
  , tsLabel             :: !(Maybe ByteString)
  }
  deriving (Eq, Ord, Show)
```

---

### Atoms and heap objects

```haskell
data Atom     -- Q: should atom fit into a cpu register? A: yes
  = HeapPtr       !Addr
  | Literal       !Lit  -- TODO: remove this
  | Void
  | PtrAtom       !PtrOrigin !(Ptr Word8)
  | IntAtom       !Int
  | WordAtom      !Word
  | FloatAtom     !Float
  | DoubleAtom    !Double
  | MVar          !Int
  | MutVar        !Int
  | Array             !ArrIdx
  | MutableArray      !ArrIdx
  | SmallArray        !SmallArrIdx
  | SmallMutableArray !SmallArrIdx
  | ArrayArray        !ArrayArrIdx
  | MutableArrayArray !ArrayArrIdx
  | ByteArray         !ByteArrayIdx
  | MutableByteArray  !ByteArrayIdx
  | WeakPointer       !Int
  | StableName        !Int
  | ThreadId          !Int
  | LiftedUndefined
  deriving (Show, Eq, Ord)
```

---

### Atoms and heap objects

```haskell
data HeapObject
  = Con
	{ hoIsLNE       :: Bool
	, hoCon         :: DataCon
	, hoConArgs     :: [Atom]
	}
  | Closure
	{ hoIsLNE       :: Bool
	, hoName        :: Id
	, hoCloBody     :: StgRhsClosure
	, hoEnv         :: Env    -- local environment ; with live variables only, everything else is pruned
	, hoCloArgs     :: [Atom]
	, hoCloMissing  :: Int    -- HINT: this is a Thunk if 0 arg is missing ; if all is missing then Fun ; Pap is some arg is provided
	}
  | BlackHole HeapObject
  | ApStack                   -- HINT: needed for the async exceptions
	{ hoResult      :: [Atom]
	, hoStack       :: [StackContinuation]
	}
  | RaiseException Atom
  deriving (Show, Eq, Ord)
```

### Blackholes

![](https://user-images.githubusercontent.com/875324/178807177-1144487a-e230-4aa7-b005-6c4cdc73567b.png)

---

### ESTG Primops

Primops dir.

```haskell
  -rw-r--r--   1 dmjio  staff   5649 Jul 12 21:59 Array.hs
  -rw-r--r--   1 dmjio  staff   6270 Jul 12 21:59 ArrayArray.hs
  -rw-r--r--   1 dmjio  staff  41019 Jul 12 21:59 ByteArray.hs
  -rw-r--r--   1 dmjio  staff   1384 Jul 12 21:59 Char.hs
  -rw-r--r--   1 dmjio  staff   6550 Jul 12 21:59 Compact.hs
  -rw-r--r--   1 dmjio  staff   9480 Jul 12 21:59 Concurrency.hs
  -rw-r--r--   1 dmjio  staff   2601 Jul 12 21:59 DelayWait.hs
  -rw-r--r--   1 dmjio  staff   4478 Jul 12 21:59 Double.hs
  -rw-r--r--   1 dmjio  staff   6129 Jul 12 21:59 Exceptions.hs
  -rw-r--r--   1 dmjio  staff   4301 Jul 12 21:59 Float.hs
  -rw-r--r--   1 dmjio  staff   3119 Jul 12 21:59 GHCiBytecode.hs
  -rw-r--r--   1 dmjio  staff   6043 Jul 12 21:59 Int.hs
  -rw-r--r--   1 dmjio  staff   2696 Jul 12 21:59 Int16.hs
  -rw-r--r--   1 dmjio  staff   2584 Jul 12 21:59 Int8.hs
  -rw-r--r--   1 dmjio  staff   8302 Jul 12 21:59 MVar.hs
  -rw-r--r--   1 dmjio  staff   8268 Jul 12 21:59 MiscEtc.hs
  -rw-r--r--   1 dmjio  staff   3121 Jul 12 21:59 MutVar.hs
  -rw-r--r--   1 dmjio  staff   1332 Jul 12 21:59 Narrowings.hs
  -rw-r--r--   1 dmjio  staff   1068 Jul 12 21:59 Parallelism.hs
  -rw-r--r--   1 dmjio  staff   2288 Jul 12 21:59 Prefetch.hs
  -rw-r--r--   1 dmjio  staff   4349 Jul 12 21:59 STM.hs
  -rw-r--r--   1 dmjio  staff   7358 Jul 12 21:59 SmallArray.hs
  -rw-r--r--   1 dmjio  staff   2131 Jul 12 21:59 StablePointer.hs
  -rw-r--r--   1 dmjio  staff   2096 Jul 12 21:59 TagToEnum.hs
  -rw-r--r--   1 dmjio  staff   2289 Jul 12 21:59 Unsafe.hs
  -rw-r--r--   1 dmjio  staff   3199 Jul 12 21:59 WeakPointer.hs
  -rw-r--r--   1 dmjio  staff  11005 Jul 12 21:59 Word.hs
  -rw-r--r--   1 dmjio  staff   2757 Jul 12 21:59 Word16.hs
  -rw-r--r--   1 dmjio  staff   2630 Jul 12 21:59 Word8.hs
```

### ESTG Primops

```haskell
evalPrimOp :: HasCallStack => Name -> [Atom] -> Type -> Maybe TyCon -> M [Atom]
evalPrimOp =
  PrimAddr.evalPrimOp $
  PrimArray.evalPrimOp $
  PrimSmallArray.evalPrimOp $
  PrimArrayArray.evalPrimOp $
  PrimByteArray.evalPrimOp $
  PrimChar.evalPrimOp $
  PrimConcurrency.evalPrimOp $
  PrimDelayWait.evalPrimOp $
  PrimParallelism.evalPrimOp $
  PrimExceptions.evalPrimOp $
  PrimFloat.evalPrimOp $
  PrimDouble.evalPrimOp $
  PrimInt16.evalPrimOp $
  PrimInt8.evalPrimOp $
  PrimInt.evalPrimOp $
  PrimMutVar.evalPrimOp $
  PrimMVar.evalPrimOp $
  PrimNarrowings.evalPrimOp $
  PrimPrefetch.evalPrimOp $
  PrimStablePointer.evalPrimOp $
  PrimWeakPointer.evalPrimOp $
  PrimWord16.evalPrimOp $
  PrimWord8.evalPrimOp $
  PrimWord.evalPrimOp $
  PrimTagToEnum.evalPrimOp $
  PrimUnsafe.evalPrimOp $
  PrimMiscEtc.evalPrimOp $
  unsupported where
	unsupported op args _t _tc = stgErrorM $ "unsupported StgPrimOp: " ++ show op ++ " args: " ++ show args
```

---

### PrimOp example

```haskell
evalPrimOp :: PrimOpEval -> Name -> [Atom] -> Type -> Maybe TyCon -> M [Atom]
evalPrimOp fallback op args t tc = case (op, args) of

  -- fork# :: a -> State# RealWorld -> (# State# RealWorld, ThreadId# #)
  ( "fork#", [ioAction, _s]) -> do
    currentTS <- getCurrentThreadState

    (newTId, newTS) <- createThread
    updateThreadState newTId $ newTS
      { tsCurrentResult   = [ioAction]
      , tsStack           = [Apply [Void], RunScheduler SR_ThreadFinished]

      -- NOTE: start blocked if the current thread is blocked
      , tsBlockExceptions = tsBlockExceptions currentTS
      , tsInterruptible   = tsInterruptible currentTS
      }

    scheduleToTheEnd newTId

    -- NOTE: context switch soon, but not immediately: we don't want every forkIO to force a context-switch.
    requestContextSwitch  -- TODO: push continuation reschedule, reason request context switch

    pure [ThreadId newTId]
```

---

### ESTG GC

```haskell
runGCSync :: [Atom] -> M ()
runGCSync localGCRoots = do
  stgState <- get
  rsData <- liftIO $ runLiveDataAnalysis localGCRoots stgState
  put $ (pruneStgState stgState rsData) {ssGCIsRunning = False}
  finalizeDeadWeakPointers (rsWeakPointers rsData)
  loadRetanerDb
  isQuiet <- gets ssIsQuiet
  unless isQuiet $ do
	liftIO $ do
	  reportRemovedData stgState rsData
	  reportAddressCounters stgState
	postGCReport
```

---

### ESTG Threads

```haskell
runScheduler :: [Atom] -> ScheduleReason -> M [Atom]
runScheduler result sr = do
  tid <- gets ssCurrentThreadId
  --liftIO $ putStrLn $ " * scheduler: " ++ show sr ++ " thread: " ++ show tid
  case sr of
	SR_ThreadFinished -> do
	  -- set thread status to finished
	  ts <- getThreadState tid
	  updateThreadState tid ts {tsStatus = ThreadFinished}
	  yield result

	SR_ThreadBlocked  -> yield result

	SR_ThreadYield    -> yield result

yield :: [Atom] -> M [Atom]
yield result = do
  tid <- gets ssCurrentThreadId
  ts <- getThreadState tid
  -- save result
  updateThreadState tid ts {tsCurrentResult = result}

  wakeUpSleepingThreads

  -- lookup next thread
  nextTid <- getNextRunnableThread
  --liftIO $ putStrLn $ " * scheduler next runnable thread: " ++ show nextTid

  -- switchToThread
  switchToThread nextTid

  -- return threads current result
  nextTS <- getThreadState nextTid

  -- TODO/IDEA/IMPROVEMENT:
  --    store this value in the thread state, but only for the suspended states
  --    the running threads should not store the old "current result" that would prevent garbage collection
  -- HINT: clear value to allow garbage collection
  updateThreadState nextTid nextTS {tsCurrentResult = []}

  pure $ tsCurrentResult nextTS

```

```haskell
waitAndScheduleBlockedThreads :: M [Int]
waitAndScheduleBlockedThreads = do
  tsList <- gets $ IntMap.toList . ssThreads
  let blockedThreads = [(tid, ts) | (tid, ts) <- tsList, isBlocked (tsStatus ts)]
	  isBlocked = \case
		ThreadBlocked{} -> True
		_ -> False
  if null blockedThreads
	then do
	  modify' $ \s -> s {ssScheduledThreadIds = []}
	  pure $ map fst tsList
	else do
	  handleBlockedDelayWait
	  calculateNewSchedule
```

### ESTG IOManager

```haskell
handleBlockedDelayWait :: M ()
handleBlockedDelayWait = do
  tsList <- gets $ IntMap.toList . fmap tsStatus . ssThreads
  now <- liftIO getCurrentTime
  let maxSeconds  = 31 * 24 * 60 * 60 -- some OS have this constraint
	  maxDelay    = secondsToNominalDiffTime maxSeconds
	  delaysT     = [(tid, t `diffUTCTime` now) | (tid, ThreadBlocked (BlockedOnDelay t)) <- tsList]
	  minDelay    = max 0 $ minimum $ maxDelay : delays
	  readFDsT    = [(tid, fromIntegral fd :: CInt) | (tid, ThreadBlocked (BlockedOnRead fd)) <- tsList]
	  writeFDsT   = [(tid, fromIntegral fd :: CInt) | (tid, ThreadBlocked (BlockedOnWrite fd)) <- tsList]
	  delays      = map snd delaysT
	  readFDs     = map snd readFDsT
	  writeFDs    = map snd writeFDsT
	  fdList      = readFDs ++ writeFDs
	  maxFD       = maximum fdList
  -- TODO: detect deadlocks
  if maxDelay == 0 then pure () else unless (null fdList) $ do
	-- query file descriptors
	(selectResult, errorNo) <- liftIO $ waitForFDs (V.fromList readFDs) (V.fromList writeFDs) maxFD
	when (selectResult < 0) $ error $ "select error, errno: " ++ show errorNo

	forM_ readFDsT $ \(tid, fd) -> do
	  liftIO (fdPollReadState fd) >>= \case
		0 -> do
		  ts <- getThreadState tid
		  updateThreadState tid ts {tsStatus = ThreadRunning}
		_ -> pure () -- TODO

	forM_ writeFDsT $ \(tid, fd) -> do
	  liftIO (fdPollWriteState fd) >>= \case
		0 -> do
		  ts <- getThreadState tid
		  updateThreadState tid ts {tsStatus = ThreadRunning}
		_ -> pure () -- TODO

	pure ()

fdPollReadState :: CInt -> IO CInt
fdPollReadState fd = do
  [C.block| int {
	int r;
	fd_set rfd;
	struct timeval now;

	FD_ZERO(&rfd);
	FD_SET($(int fd), &rfd);

	/* only poll */
	now.tv_sec  = 0;
	now.tv_usec = 0;
	for (;;)
	{
		r = select( $(int fd) + 1, &rfd, NULL, NULL, &now);
		/* the descriptor is sane */
		if (r != -1)
			break;

		switch (errno)
		{
			case EBADF: return 2; //RTS_FD_IS_INVALID
			case EINTR: continue;
			default:    return 3; //RTS_SELECT_FAILURE
		}
	}

	if (r == 0)
		return 1; //RTS_FD_IS_BLOCKING
	else
		return 0; //RTS_FD_IS_READY
  } |]

```

---

### ESTG FFI

```haskell
evalForeignCall :: FunPtr a -> [FFI.Arg] -> Type -> IO [Atom]
evalForeignCall funPtr cArgs retType = case retType of
  UnboxedTuple [] -> do
	_result <- FFI.callFFI funPtr FFI.retVoid cArgs
	pure []
  -- ...

```

```haskell
getFFISymbol :: Name -> M (FunPtr a)
getFFISymbol name = do
  dl <- gets ssCBitsMap
  funPtr <- liftIO . BS8.useAsCString name $ c_dlsym (packDL dl)
  case funPtr == nullFunPtr of
    False -> pure funPtr
    True  -> if Set.member name rtsSymbolSet
      then stgErrorM $ "this RTS symbol is not implemented yet: " ++ BS8.unpack name
      else stgErrorM $ "unknown foreign symbol: " ++ BS8.unpack name
```

---

### ESTG Apply

```haskell
builtinStgApply :: HasCallStack => StaticOrigin -> Atom -> [Atom] -> M [Atom]
builtinStgApply so a [] = builtinStgEval so a
builtinStgApply so a@HeapPtr{} args = do
  let argCount      = length args
	  HeapPtr addr  = a
  o <- readHeap a
  case o of
	RaiseException ex -> PrimExceptions.raiseEx ex
	Con{}             -> stgErrorM $ "unexpexted con at apply: "-- ++ show o
	BlackHole t       -> stgErrorM $ "blackhole ; loop in application of : " ++ show t
	Closure{..}
	  -- under saturation
	  | hoCloMissing - argCount > 0
	  -> do
		newAp <- freshHeapAddress
		store newAp (o {hoCloArgs = hoCloArgs ++ args, hoCloMissing = hoCloMissing - argCount})
		pure [HeapPtr newAp]

	  -- over saturation
	  | hoCloMissing - argCount < 0
	  -> do
		let (satArgs, remArgs) = splitAt hoCloMissing args
		stackPush (Apply remArgs)
		stackPushRestoreProgramPoint $ length remArgs -- HINT: for call-graph builder ; use the current closure as call origin
		builtinStgApply so a satArgs

	  -- saturation
	  | hoCloMissing - argCount == 0
	  -> do
		newAp <- freshHeapAddress
		store newAp (o {hoCloArgs = hoCloArgs ++ args, hoCloMissing = hoCloMissing - argCount})
		builtinStgEval so (HeapPtr newAp)

builtinStgApply so a args = stgErrorM $ "builtinStgApply - expected a closure (ptr), got: " ++
  show a ++ ", args: " ++ show args ++ ", static-origin: " ++ show so
```

---

### Eval

```haskell
builtinStgEval :: HasCallStack => StaticOrigin -> Atom -> M [Atom]
builtinStgEval so a@HeapPtr{} = do
  o <- readHeap a
  case o of
	RaiseException ex -> PrimExceptions.raiseEx ex
	Con{}       -> pure [a]
	{-
	-- TODO: check how the cmm stg machine handles this case
	BlackHole t -> do
					Rts{..} <- gets ssRtsSupport
					liftIO $ do
					  hPutStrLn stderr $ takeBaseName rtsProgName ++ ": <<loop>>"
					  exitWith ExitSuccess
					stgErrorM $ "blackhole ; loop in evaluation of : " ++ show t
	-}
	Closure{..}
	  | hoCloMissing /= 0
	  -> pure [a]

	  | otherwise
	  -> do

		let StgRhsClosure _ uf params e = hoCloBody
			HeapPtr l = a
			extendedEnv = addManyBindersToEnv SO_CloArg params hoCloArgs hoEnv

		modify' $ \s@StgState{..} -> s {ssClosureCallCounter = succ ssClosureCallCounter}
		markExecuted l
		markExecutedId hoName

		-- build call graph
		buildCallGraph so hoName

		modify' $ \s -> s {ssCurrentClosure = Just hoName, ssCurrentClosureEnv = extendedEnv, ssCurrentClosureAddr = l}
		-- check breakpoints and region entering
		let closureName = binderUniqueName $ unId hoName
		markClosure closureName -- HINT: this list can be deleted by a debugger command, so this is not the same as `markExecutedId`
		Debugger.checkBreakpoint closureName
		Debugger.checkRegion closureName
		GC.checkGC [a] -- HINT: add local env as GC root

		-- TODO: env or free var handling
		case uf of
		  ReEntrant -> do
			-- closure may be entered multiple times, but should not be updated or blackholed.
			evalExpr extendedEnv e
		  Updatable -> do
			-- closure should be updated after evaluation (and may be blackholed during evaluation).
			stackPush (Update l)
			store l (BlackHole o)
			evalExpr extendedEnv e
		  SingleEntry -> do
			-- closure will only be entered once, and so need not be updated but may safely be blackholed.
			store l (BlackHole o)
			evalExpr extendedEnv e
	_ -> stgErrorM $ "expected heap object: " ++ show o
builtinStgEval so a = stgErrorM $ "expected a thunk, got: " ++ show a ++ ", static-origin: " ++ show so
```

-- A thunk is a closure with all of its arguments satisfied.

---

### Eval STG

```haskell
evalExpr :: HasCallStack => Env -> Expr -> M [Atom]
evalExpr localEnv = \case
  StgTick _ e       -> evalExpr localEnv e
  StgLit l          -> pure <$> evalLiteral l
  StgConApp dc l _
	-- HINT: make and return unboxed tuple
	| UnboxedTupleCon{} <- dcRep dc
	-> mapM (evalArg localEnv) l   -- Q: is this only for unboxed tuple? could datacon be heap allocated?

	-- HINT: create boxed datacon on the heap
	| otherwise
	-> do
	  args <- mapM (evalArg localEnv) l
	  loc <- allocAndStore (Con False dc args)
	  pure [HeapPtr loc]

  StgLet b e -> do
	extendedEnv <- declareBinding False localEnv b
	evalExpr extendedEnv e

  StgLetNoEscape b e -> do -- TODO: do not allocate closure on heap, instead put into env (stack) allocated closure ; model stack allocated heap objects
	extendedEnv <- declareBinding True localEnv b
	evalExpr extendedEnv e

  -- var (join id)
  StgApp i [] _t _
	| JoinId 0 <- binderDetails i
	-> do
	  -- HINT: join id-s are always closures, needs eval
	  -- NOTE: join id's type tells the closure return value representation
	  (so, v) <- lookupEnvSO localEnv i
	  builtinStgEval so v

	| JoinId x <- binderDetails i
	-> stgErrorM $ "join-id var arity error, expected 0, got: " ++ show x ++ " id: " ++ show i

  -- var (non join id)
  StgApp i [] _t _ -> case binderType i of

	SingleValue LiftedRep -> do
	  -- HINT: must be HeapPtr ; read heap ; check if Con or Closure ; eval if Closure ; return HeapPtr if Con
	  (so, v) <- lookupEnvSO localEnv i
	  builtinStgEval so v

	SingleValue _ -> do
	  v <- lookupEnv localEnv i
	  pure [v]

	UnboxedTuple []
	  | binderUniqueName i == "ghc-prim_GHC.Prim.coercionToken#" -- wired in coercion token ; FIXME: handle wired-in names with a better design
	  -> do
		pure []

	r -> stgErrorM $ "unsupported var rep: " ++ show r ++ " " ++ show i -- unboxed: is it possible??

  -- fun app
  --  Q: should app always be lifted/unlifted?
  --  Q: what does unlifted app mean? (i.e. no Ap node, but saturated calls to known functions only?)
  --  A: the join id type is for the return value representation and not for the id representation, so it can be unlifted.
  StgApp i l _t _
	| JoinId _ <- binderDetails i
	-> do
	  args <- mapM (evalArg localEnv) l
	  (so, v) <- lookupEnvSO localEnv i
	  builtinStgApply so v args

  {- non-join id -}
  StgApp i l _t _ -> case binderType i of
	SingleValue LiftedRep -> do
	  args <- mapM (evalArg localEnv) l
	  (so, v) <- lookupEnvSO localEnv i
	  builtinStgApply so v args

	r -> stgErrorM $ "unsupported app rep: " ++ show r -- unboxed: invalid

  StgCase e scrutineeResult altType alts -> do
	Just curClosure <- gets ssCurrentClosure
	curClosureAddr <- gets ssCurrentClosureAddr
	stackPush (CaseOf curClosureAddr curClosure localEnv scrutineeResult altType alts)
	setProgramPoint . PP_Scrutinee $ Id scrutineeResult
	evalExpr localEnv e

  StgOpApp (StgPrimOp op) l t tc -> do
	Debugger.checkBreakpoint op
	Debugger.checkRegion op
	markPrimOp op
	args <- mapM (evalArg localEnv) l
	tid <- gets ssCurrentThreadId
	evalPrimOp op args t tc

  StgOpApp (StgFCallOp foreignCall) l t tc -> do
	-- check foreign target region and breakpoint
	case foreignCTarget foreignCall of
	  StaticTarget _ targetName _ _ -> do
		Debugger.checkBreakpoint targetName
		Debugger.checkRegion targetName
	  _ -> pure ()

markFFI foreignCall
	args <- case foreignCTarget foreignCall of
	  StaticTarget _ "createAdjustor" _ _
		| [arg0, arg1, arg2, arg3, arg4, arg5] <- l
		-> do
			-- HINT: do not resolve the unused label pointer that comes from the stub code
			mapM (evalArg localEnv) [arg0, arg1, StgLitArg LitNullAddr, arg3, arg4, arg5]
	  _ -> mapM (evalArg localEnv) l
	evalFCallOp evalOnNewThread foreignCall args t tc

  StgOpApp (StgPrimCallOp primCall) l t tc -> do
	markPrimCall primCall
	args <- mapM (evalArg localEnv) l
	evalPrimCallOp primCall args t tc

  StgOpApp op _args t _tc -> stgErrorM $ "unsupported StgOp: " ++ show op ++ " :: " ++ show t
```

---

### GRIN vs. STG

- Evaluation of a suspended function application
  - STG: The function to call is unknown at compile time, the evaluation of a closure will result in some kind of indirect call, through a pointer. This makes optimisations harder.
  - GRIN: forbids all unknown calls and uses control flow analysis to approximate the possible targets.

---

### Thanks!
![](https://upload.wikimedia.org/wikipedia/commons/b/bc/Face-grin.svg)
