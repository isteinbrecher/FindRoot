#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Text:: *)
(*Solution for test cases*)


f[x_]:={
x[[1]]^2+x[[2]]+2,
6*x[[1]]-x[[2]]*(x[[1]]+1)
}

FindRoot[f[{x1,x2}],{{x1,0},{x2,0}}]
