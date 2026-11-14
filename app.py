"""
app.py - Production-Ready Interactive Web Application
Four Worlds Adversarial Fraud Detection Laboratory

Connects the refined Tailwind UI directly to live XGBoost + LSTM models
with real-time inference, presets, and metric telemetry.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import time

from src.models.xgboost_detector import XGBoostDetector
from src.models.lstm_detector import LSTMDetector
from src.config import FEATURE_COLUMNS, OUTPUT_DIR, MODELS_DIR

PORT = 5000

print("=" * 70)
print("  Initializing FraudShield AI Detection Backend...")
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

xgb_model = XGBoostDetector()
xgb_model.load("xgboost_detector.json")

lstm_model = LSTMDetector()
lstm_model.load("lstm_detector.pt")

print(f"  [OK] Models Loaded (XGBoost Threshold: {xgb_model.threshold:.3f}, LSTM Threshold: {lstm_model.sim_threshold:.3f})")
print("=" * 70)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <title>FraudShield AI - Live Adversarial Payment Inspector</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
  <script src="https://cdn.tailwindcss.com?plugins=forms"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ["Plus Jakarta Sans", "Inter", "sans-serif"],
            mono: ["JetBrains Mono", "monospace"]
          }
        }
      }
    };
  </script>
  <style>
    body { background-color: #faf9f3; color: #18181b; }
    .no-scrollbar::-webkit-scrollbar { display: none; }
    .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
  </style>
</head>
<body class="font-sans antialiased min-h-screen pb-16">

  <!-- TOP APP BAR -->
  <header class="sticky top-0 z-50 bg-[#faf9f3]/95 backdrop-blur-md border-b border-[#e7e5dc] px-4 py-3">
    <div class="max-w-6xl mx-auto flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-neutral-900 text-white flex items-center justify-center font-bold shadow-sm">
          <span class="material-symbols-outlined text-[24px]">shield</span>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-bold text-lg text-neutral-900 tracking-tight">FraudShield AI</span>
            <span class="font-mono text-[10px] text-neutral-700 font-semibold px-2 py-0.5 rounded bg-neutral-200 border border-neutral-300">Four Worlds Lab</span>
          </div>
          <span class="font-mono text-[11px] text-neutral-500 font-medium tracking-wide uppercase">Adversarial Robustness Inspector</span>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <div class="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white border border-[#e7e5dc] shadow-sm">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span class="text-xs font-semibold text-neutral-800">Dual AI Online</span>
        </div>
        <div class="flex items-center gap-1 px-3 py-1 rounded-full bg-white border border-[#e7e5dc] shadow-sm font-mono text-xs text-neutral-700">
          <span class="text-amber-500">⚡</span>
          <span id="headerLatency">14ms</span>
        </div>
      </div>
    </div>
  </header>

  <!-- MAIN CONTAINER -->
  <main class="max-w-6xl mx-auto px-4 pt-6 space-y-6">

    <!-- ATTACK SCENARIO QUICK DEMO BAR -->
    <section class="space-y-2">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <span class="w-2 h-2 rounded-full bg-neutral-900"></span>
          <span class="text-xs font-bold uppercase tracking-wider text-neutral-500">Preset Scenarios (Click to test instantly)</span>
        </div>
        <span class="font-mono text-xs text-neutral-500 font-medium">5 Ready Vectors</span>
      </div>
      <div class="flex items-center space-x-2 overflow-x-auto no-scrollbar py-1" id="presetBar">
        <button class="flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-neutral-900 text-white shadow-sm border border-neutral-800 transition-all hover:opacity-90" id="btn-mule" onclick="applyPreset('mule')">
          <span class="text-rose-400 text-xs">●</span>
          <span class="font-mono text-xs font-semibold">Mule Account Drain</span>
          <span class="font-mono text-xs opacity-75">(₹5,00,000)</span>
        </button>
        <button class="flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-white border border-[#e7e5dc] text-neutral-700 hover:text-neutral-900 shadow-sm transition-all" id="btn-smurfing" onclick="applyPreset('smurfing')">
          <span class="text-amber-500 text-xs">●</span>
          <span class="font-mono text-xs">Smurfing Evasion</span>
          <span class="font-mono text-xs text-neutral-400">(₹9,500)</span>
        </button>
        <button class="flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-white border border-[#e7e5dc] text-neutral-700 hover:text-neutral-900 shadow-sm transition-all" id="btn-cashout" onclick="applyPreset('cashout')">
          <span class="text-rose-400 text-xs">●</span>
          <span class="font-mono text-xs">Rapid CASH_OUT Drain</span>
        </button>
        <button class="flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-white border border-[#e7e5dc] text-neutral-700 hover:text-neutral-900 shadow-sm transition-all" id="btn-grocery" onclick="applyPreset('grocery')">
          <span class="text-emerald-500 text-xs">●</span>
          <span class="font-mono text-xs">Grocery POS</span>
          <span class="font-mono text-xs text-neutral-400">(₹650)</span>
        </button>
        <button class="flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-white border border-[#e7e5dc] text-neutral-700 hover:text-neutral-900 shadow-sm transition-all" id="btn-rent" onclick="applyPreset('rent')">
          <span class="text-emerald-500 text-xs">●</span>
          <span class="font-mono text-xs">Rent Payment</span>
          <span class="font-mono text-xs text-neutral-400">(₹25,000)</span>
        </button>
      </div>
    </section>

    <!-- MAIN TWO-COLUMN DASHBOARD -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

      <!-- LEFT COLUMN: TRANSACTION PARAMETERS SIMULATOR (7 cols) -->
      <section class="lg:col-span-7 bg-white p-6 rounded-2xl border border-[#e7e5dc] shadow-sm space-y-5">
        <div class="flex items-center justify-between border-b border-[#e7e5dc] pb-3">
          <div class="flex items-center space-x-2">
            <span class="material-symbols-outlined text-neutral-900 text-[20px]">tune</span>
            <h2 class="text-base font-bold text-neutral-900">Transaction Parameters</h2>
          </div>
          <span class="font-mono text-xs text-neutral-500">UPI / PaySim Framing</span>
        </div>

        <form id="txForm" onsubmit="runInspection(event)" class="space-y-4">
          <!-- TX Type Selector -->
          <div>
            <label class="text-[11px] font-bold text-neutral-500 uppercase tracking-wider block mb-1.5">Transaction Type</label>
            <div class="grid grid-cols-3 gap-2 bg-[#faf9f3] p-1.5 rounded-xl border border-[#e7e5dc]">
              <button type="button" class="py-2 rounded-lg text-center font-mono text-xs bg-neutral-900 text-white font-semibold shadow-sm transition-all" id="type-transfer" onclick="selectType('TRANSFER')">TRANSFER</button>
              <button type="button" class="py-2 rounded-lg text-center font-mono text-xs text-neutral-600 hover:text-neutral-900 transition-all" id="type-cashout" onclick="selectType('CASH_OUT')">CASH_OUT</button>
              <button type="button" class="py-2 rounded-lg text-center font-mono text-xs text-neutral-600 hover:text-neutral-900 transition-all" id="type-payment" onclick="selectType('PAYMENT')">PAYMENT</button>
            </div>
            <input type="hidden" id="txType" value="TRANSFER"/>
          </div>

          <!-- Amount -->
          <div>
            <div class="flex items-center justify-between mb-1">
              <label class="text-[11px] font-bold text-neutral-500 uppercase tracking-wider">Transfer Amount</label>
              <span class="font-mono text-xs text-neutral-500">INR (₹)</span>
            </div>
            <div class="relative flex items-center">
              <span class="absolute left-3.5 font-mono text-base font-bold text-neutral-400">₹</span>
              <input class="w-full bg-[#faf9f3] text-neutral-900 font-mono text-base font-bold py-2.5 pl-9 pr-3 rounded-xl border border-[#e7e5dc] focus:outline-none focus:border-neutral-900 focus:bg-white transition-all shadow-inner" id="amount" oninput="recalcBalances()" type="number" step="any" required value="500000"/>
            </div>
          </div>

          <!-- Balances Grid -->
          <div class="grid grid-cols-2 gap-3">
            <div class="p-3 rounded-xl bg-[#faf9f3] border border-[#e7e5dc] space-y-2">
              <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider block">Sender (Origin Account)</span>
              <div>
                <label class="text-[10px] text-neutral-500 block">Old Balance (₹)</label>
                <input class="w-full bg-white font-mono text-xs font-semibold p-2 rounded-lg border border-[#e7e5dc]" id="oldbalanceOrg" oninput="recalcBalances()" type="number" step="any" required value="500000"/>
              </div>
              <div>
                <label class="text-[10px] text-neutral-500 block">New Balance (₹)</label>
                <input class="w-full bg-white font-mono text-xs font-semibold p-2 rounded-lg border border-[#e7e5dc]" id="newbalanceOrig" type="number" step="any" required value="0"/>
              </div>
            </div>

            <div class="p-3 rounded-xl bg-[#faf9f3] border border-[#e7e5dc] space-y-2">
              <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider block">Receiver (Beneficiary)</span>
              <div>
                <label class="text-[10px] text-neutral-500 block">Old Balance (₹)</label>
                <input class="w-full bg-white font-mono text-xs font-semibold p-2 rounded-lg border border-[#e7e5dc]" id="oldbalanceDest" oninput="recalcBalances()" type="number" step="any" required value="0"/>
              </div>
              <div>
                <label class="text-[10px] text-neutral-500 block">New Balance (₹)</label>
                <input class="w-full bg-white font-mono text-xs font-semibold p-2 rounded-lg border border-[#e7e5dc]" id="newbalanceDest" type="number" step="any" required value="500000"/>
              </div>
            </div>
          </div>

          <!-- Sliders and Toggles -->
          <div class="grid grid-cols-2 gap-3">
            <div class="p-3 rounded-xl bg-[#faf9f3] border border-[#e7e5dc] space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Velocity Count</span>
                <span class="font-mono text-xs font-bold text-neutral-900" id="txCountVal">1 Tx</span>
              </div>
              <input class="w-full h-1.5 bg-[#e7e5dc] rounded-lg appearance-none cursor-pointer accent-neutral-900 mt-2" id="tx_count_last_N" max="10" min="1" oninput="document.getElementById('txCountVal').innerText = this.value + ' Tx'" type="range" value="1"/>
            </div>

            <div class="p-3 rounded-xl bg-[#faf9f3] border border-[#e7e5dc] space-y-1">
              <div class="flex items-center justify-between">
                <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Timing Delay</span>
                <span class="font-mono text-xs font-bold text-neutral-900" id="delayVal">999 Hours</span>
              </div>
              <input class="w-full h-1.5 bg-[#e7e5dc] rounded-lg appearance-none cursor-pointer accent-neutral-900 mt-2" id="steps_since_last_tx" max="999" min="0.5" oninput="document.getElementById('delayVal').innerText = this.value + ' Hours'" type="range" value="999"/>
            </div>
          </div>

          <!-- Beneficiary Novelty -->
          <div class="flex items-center justify-between p-3 rounded-xl bg-[#faf9f3] border border-[#e7e5dc]">
            <div class="flex flex-col">
              <span class="text-xs font-bold text-neutral-900">Beneficiary History</span>
              <span class="text-[11px] text-neutral-500">First-time transfer or established contact</span>
            </div>
            <select id="is_new_beneficiary" class="bg-white font-mono text-xs font-bold py-1.5 px-3 rounded-lg border border-[#e7e5dc] focus:outline-none">
              <option value="1">New Beneficiary (0 History)</option>
              <option value="0">Trusted Contact (Established)</option>
            </select>
          </div>

          <!-- Submit Button -->
          <button type="submit" class="w-full flex items-center justify-center space-x-2 py-3.5 px-4 rounded-xl bg-neutral-900 hover:bg-neutral-800 text-white font-bold text-sm shadow-md transition-all active:scale-[0.99]" id="btnInspect">
            <span class="material-symbols-outlined text-[20px]">play_arrow</span>
            <span>Run Dual AI Fraud Inspection</span>
          </button>
        </form>
      </section>

      <!-- RIGHT COLUMN: LIVE VERDICT & EXPLAINABILITY (5 cols) -->
      <section class="lg:col-span-5 space-y-4">

        <!-- Verdict Banner Card -->
        <div class="rounded-2xl bg-white border border-neutral-300 p-5 shadow-sm transition-all duration-300" id="verdictBanner">
          <div class="flex items-start gap-3.5">
            <div class="w-12 h-12 rounded-xl bg-neutral-100 border border-neutral-300 flex items-center justify-center text-neutral-900 flex-shrink-0" id="verdictIconBox">
              <span class="material-symbols-outlined text-[26px]" id="verdictIcon">shield</span>
            </div>
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <span class="font-bold text-lg text-neutral-900 tracking-tight" id="verdictHeadline">FRAUD DETECTED</span>
                <span class="px-2 py-0.5 rounded-full bg-neutral-900 text-white text-[10px] font-bold uppercase tracking-wide" id="verdictTag">BLOCKED</span>
              </div>
              <p class="text-xs text-neutral-600 mt-1" id="verdictSubtext">Transaction intercepted to prevent unauthorized account drain.</p>
            </div>
          </div>

          <!-- Telemetry Bar -->
          <div class="mt-4 grid grid-cols-3 gap-2 bg-[#faf9f3] p-3 rounded-xl border border-[#e7e5dc]">
            <div class="flex flex-col">
              <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Gate Status</span>
              <span class="font-mono text-xs text-neutral-900 font-semibold mt-0.5" id="gateStatus">Intercepted</span>
            </div>
            <div class="flex flex-col">
              <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Risk Level</span>
              <span class="font-mono text-xs text-neutral-900 font-bold mt-0.5" id="riskLevel">High Risk</span>
            </div>
            <div class="flex flex-col">
              <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">AI Latency</span>
              <span class="font-mono text-xs text-emerald-700 font-semibold mt-0.5" id="boxLatency">15.4ms</span>
            </div>
          </div>
        </div>

        <!-- Dual Model Score Cards -->
        <div class="grid grid-cols-2 gap-3">
          <!-- XGBoost -->
          <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">XGBoost</span>
                <span class="px-1.5 py-0.5 rounded text-[10px] font-bold font-mono" id="xgbBadge">FLAGGED</span>
              </div>
              <div class="text-xs text-neutral-600">Tabular snapshot</div>
            </div>
            <div class="my-3">
              <div class="font-mono text-2xl font-bold text-neutral-900" id="xgbScore">72.3%</div>
              <div class="w-full bg-[#efeee8] h-1.5 rounded-full mt-2 overflow-hidden">
                <div class="bg-neutral-900 h-full rounded-full transition-all duration-500" id="xgbBar" style="width: 72.3%;"></div>
              </div>
            </div>
            <span class="font-mono text-[11px] text-neutral-500">Threshold: 0.500</span>
          </div>

          <!-- LSTM -->
          <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-1.5">
                <span class="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">LSTM Network</span>
                <span class="px-1.5 py-0.5 rounded text-[10px] font-bold font-mono" id="lstmBadge">FLAGGED</span>
              </div>
              <div class="text-xs text-neutral-600">Sequential behavior</div>
            </div>
            <div class="my-3">
              <div class="font-mono text-2xl font-bold text-neutral-900" id="lstmScore">75.0%</div>
              <div class="w-full bg-[#efeee8] h-1.5 rounded-full mt-2 overflow-hidden">
                <div class="bg-neutral-900 h-full rounded-full transition-all duration-500" id="lstmBar" style="width: 75.0%;"></div>
              </div>
            </div>
            <span class="font-mono text-[11px] text-neutral-500">Threshold: 0.325</span>
          </div>
        </div>

        <!-- Explainability / Behavioral Risk Factors -->
        <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-[11px] font-bold text-neutral-500 uppercase tracking-wider">Behavioral Attribution</span>
            <span class="font-mono text-[11px] text-neutral-500 font-medium">Explainable AI</span>
          </div>
          <div class="space-y-1.5" id="factorsList">
            <div class="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-neutral-100 text-neutral-900 border border-neutral-200 text-xs">
              <span class="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-neutral-200">DRAIN</span>
              <span>100% balance depletion detected</span>
            </div>
            <div class="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#faf9f3] text-neutral-800 border border-[#e7e5dc] text-xs">
              <span class="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-neutral-200">NOVEL DEST</span>
              <span>Beneficiary has zero prior interaction</span>
            </div>
          </div>
        </div>

      </section>
    </div>

    <!-- FOUR WORLDS ADVERSARIAL FRAMEWORK RESEARCH CARDS -->
    <section class="space-y-3 pt-2">
      <div class="flex items-center justify-between border-b border-[#e7e5dc] pb-2">
        <div class="flex items-center space-x-2">
          <span class="material-symbols-outlined text-neutral-900 text-[20px]">hub</span>
          <h2 class="text-base font-bold text-neutral-900">Four Worlds Adversarial Research Framework</h2>
        </div>
        <span class="font-mono text-xs text-neutral-600 bg-white px-2 py-0.5 rounded-full border border-[#e7e5dc]">PaySim Evaluation</span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
        <!-- W1 -->
        <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="px-2 py-0.5 rounded-md bg-[#faf9f3] border border-[#e7e5dc] font-mono text-[11px] text-neutral-900 font-bold">W1</span>
            <span class="font-mono text-[11px] text-emerald-700 font-bold">99.2% Test F1</span>
          </div>
          <div class="font-bold text-sm text-neutral-900">Known Static</div>
          <p class="text-xs text-neutral-500 leading-relaxed">Standard supervised training on familiar historical fraud archetypes.</p>
        </div>

        <!-- W2 -->
        <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="px-2 py-0.5 rounded-md bg-neutral-100 border border-neutral-300 font-mono text-[11px] text-neutral-900 font-bold">W2</span>
            <span class="font-mono text-[11px] text-neutral-900 font-bold">PPO Adaptive</span>
          </div>
          <div class="font-bold text-sm text-neutral-900">Known Adaptive</div>
          <p class="text-xs text-neutral-500 leading-relaxed">RL agent adjusts amounts and timing to evade static thresholds.</p>
        </div>

        <!-- W3 -->
        <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="px-2 py-0.5 rounded-md bg-[#faf9f3] border border-[#e7e5dc] font-mono text-[11px] text-neutral-900 font-bold">W3</span>
            <span class="font-mono text-[11px] text-neutral-700 font-medium">Unseen Patterns</span>
          </div>
          <div class="font-bold text-sm text-neutral-900">Unknown Static</div>
          <p class="text-xs text-neutral-500 leading-relaxed">Zero-day attack patterns (slow drip micro-drain, interleaved noise).</p>
        </div>

        <!-- W4 -->
        <div class="p-4 rounded-2xl bg-white border border-[#e7e5dc] shadow-sm space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="px-2 py-0.5 rounded-md bg-rose-50 border border-rose-200 font-mono text-[11px] text-rose-700 font-bold">W4</span>
            <span class="font-mono text-[11px] text-rose-700 font-bold">DD = 0.163</span>
          </div>
          <div class="font-bold text-sm text-neutral-900">Unknown Adaptive</div>
          <p class="text-xs text-neutral-500 leading-relaxed">Most challenging world: adversarial evolution over novel fraud archetypes.</p>
        </div>
      </div>
    </section>

  </main>

  <script>
    const presets = {
      mule: {
        type: 'TRANSFER',
        amount: 500000,
        oldbalanceOrg: 500000,
        newbalanceOrig: 0,
        oldbalanceDest: 0,
        newbalanceDest: 500000,
        tx_count_last_N: 1,
        steps_since_last_tx: 999,
        is_new_beneficiary: '1'
      },
      smurfing: {
        type: 'TRANSFER',
        amount: 9500,
        oldbalanceOrg: 50000,
        newbalanceOrig: 40500,
        oldbalanceDest: 1000,
        newbalanceDest: 10500,
        tx_count_last_N: 6,
        steps_since_last_tx: 0.5,
        is_new_beneficiary: '1'
      },
      cashout: {
        type: 'CASH_OUT',
        amount: 320000,
        oldbalanceOrg: 320000,
        newbalanceOrig: 0,
        oldbalanceDest: 5000,
        newbalanceDest: 325000,
        tx_count_last_N: 2,
        steps_since_last_tx: 1.0,
        is_new_beneficiary: '1'
      },
      grocery: {
        type: 'PAYMENT',
        amount: 650,
        oldbalanceOrg: 25000,
        newbalanceOrig: 24350,
        oldbalanceDest: 150000,
        newbalanceDest: 150650,
        tx_count_last_N: 1,
        steps_since_last_tx: 48,
        is_new_beneficiary: '0'
      },
      rent: {
        type: 'TRANSFER',
        amount: 25000,
        oldbalanceOrg: 85000,
        newbalanceOrig: 60000,
        oldbalanceDest: 42000,
        newbalanceDest: 67000,
        tx_count_last_N: 1,
        steps_since_last_tx: 720,
        is_new_beneficiary: '0'
      }
    };

    function selectType(t) {
      document.getElementById('txType').value = t;
      const types = ['TRANSFER', 'CASH_OUT', 'PAYMENT'];
      const map = {'TRANSFER': 'type-transfer', 'CASH_OUT': 'type-cashout', 'PAYMENT': 'type-payment'};
      types.forEach(item => {
        const el = document.getElementById(map[item]);
        if (item === t) {
          el.className = 'py-2 rounded-lg text-center font-mono text-xs bg-neutral-900 text-white font-semibold shadow-sm transition-all';
        } else {
          el.className = 'py-2 rounded-lg text-center font-mono text-xs text-neutral-600 hover:text-neutral-900 transition-all';
        }
      });
      recalcBalances();
    }

    function recalcBalances() {
      const amt = parseFloat(document.getElementById('amount').value) || 0;
      const oldOrg = parseFloat(document.getElementById('oldbalanceOrg').value) || 0;
      const oldDest = parseFloat(document.getElementById('oldbalanceDest').value) || 0;

      // Auto update new balances realistically
      document.getElementById('newbalanceOrig').value = Math.max(0, oldOrg - amt);
      document.getElementById('newbalanceDest').value = oldDest + amt;
    }

    function applyPreset(key) {
      const p = presets[key];
      if (!p) return;

      selectType(p.type);
      document.getElementById('amount').value = p.amount;
      document.getElementById('oldbalanceOrg').value = p.oldbalanceOrg;
      document.getElementById('newbalanceOrig').value = p.newbalanceOrig;
      document.getElementById('oldbalanceDest').value = p.oldbalanceDest;
      document.getElementById('newbalanceDest').value = p.newbalanceDest;
      document.getElementById('tx_count_last_N').value = p.tx_count_last_N;
      document.getElementById('txCountVal').innerText = p.tx_count_last_N + ' Tx';
      document.getElementById('steps_since_last_tx').value = p.steps_since_last_tx;
      document.getElementById('delayVal').innerText = p.steps_since_last_tx + ' Hours';
      document.getElementById('is_new_beneficiary').value = p.is_new_beneficiary;

      // Update button active styling
      ['mule', 'smurfing', 'cashout', 'grocery', 'rent'].forEach(name => {
        const b = document.getElementById('btn-' + name);
        if (name === key) {
          b.className = 'flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-neutral-900 text-white shadow-sm border border-neutral-800 transition-all';
        } else {
          b.className = 'flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-2 rounded-full bg-white border border-[#e7e5dc] text-neutral-700 hover:text-neutral-900 shadow-sm transition-all';
        }
      });

      // Submit immediately
      document.getElementById('txForm').requestSubmit();
    }

    async function runInspection(e) {
      if (e) e.preventDefault();
      const btn = document.getElementById('btnInspect');
      const origHtml = btn.innerHTML;
      btn.innerHTML = `<span class="material-symbols-outlined text-[20px] animate-spin">refresh</span><span>Evaluating AI Models...</span>`;

      const payload = {
        type: document.getElementById('txType').value,
        amount: parseFloat(document.getElementById('amount').value) || 0,
        oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value) || 0,
        newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value) || 0,
        oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value) || 0,
        newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value) || 0,
        tx_count_last_N: parseInt(document.getElementById('tx_count_last_N').value) || 1,
        steps_since_last_tx: parseFloat(document.getElementById('steps_since_last_tx').value) || 999,
        is_new_beneficiary: parseInt(document.getElementById('is_new_beneficiary').value) || 1,
        payer_id: 'C' + Math.floor(100000000 + Math.random() * 900000000)
      };

      const t0 = performance.now();
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        const latency = (performance.now() - t0).toFixed(1) + 'ms';
        document.getElementById('headerLatency').innerText = latency;
        document.getElementById('boxLatency').innerText = latency;
        renderVerdict(data);
      } catch (err) {
        alert('Inspection request failed: ' + err);
      } finally {
        btn.innerHTML = origHtml;
      }
    }

    function renderVerdict(data) {
      const banner = document.getElementById('verdictBanner');
      const iconBox = document.getElementById('verdictIconBox');
      const icon = document.getElementById('verdictIcon');
      const headline = document.getElementById('verdictHeadline');
      const tag = document.getElementById('verdictTag');
      const subtext = document.getElementById('verdictSubtext');
      const gateStatus = document.getElementById('gateStatus');
      const riskLevel = document.getElementById('riskLevel');

      if (data.is_fraud) {
        banner.className = 'rounded-2xl bg-white border border-rose-300 p-5 shadow-sm transition-all duration-300';
        iconBox.className = 'w-12 h-12 rounded-xl bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-700 flex-shrink-0';
        icon.innerText = 'shield';
        headline.innerText = 'FRAUD INTERCEPTED';
        headline.className = 'font-bold text-lg text-rose-700 tracking-tight';
        tag.className = 'px-2 py-0.5 rounded-full bg-rose-600 text-white text-[10px] font-bold uppercase tracking-wide';
        tag.innerText = 'BLOCKED';
        subtext.innerText = 'Transaction intercepted and blocked to prevent capital flight.';
        gateStatus.innerText = 'Dual Gate Rejection';
        gateStatus.className = 'font-mono text-xs text-rose-700 font-semibold mt-0.5';
        riskLevel.innerText = 'Critical Threat';
        riskLevel.className = 'font-mono text-xs text-rose-700 font-bold mt-0.5';
      } else {
        banner.className = 'rounded-2xl bg-white border border-emerald-300 p-5 shadow-sm transition-all duration-300';
        iconBox.className = 'w-12 h-12 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700 flex-shrink-0';
        icon.innerText = 'verified_user';
        headline.innerText = 'CLEARED AS LEGITIMATE';
        headline.className = 'font-bold text-lg text-emerald-700 tracking-tight';
        tag.className = 'px-2 py-0.5 rounded-full bg-emerald-600 text-white text-[10px] font-bold uppercase tracking-wide';
        tag.innerText = 'PASSED';
        subtext.innerText = 'Payment profile matches standard account spending behavior.';
        gateStatus.innerText = 'Cleared Both Gates';
        gateStatus.className = 'font-mono text-xs text-emerald-700 font-semibold mt-0.5';
        riskLevel.innerText = 'Low / Nominal';
        riskLevel.className = 'font-mono text-xs text-emerald-700 font-bold mt-0.5';
      }

      // Model Scores
      const xgbPct = (data.xgb_prob * 100).toFixed(1) + '%';
      document.getElementById('xgbScore').innerText = xgbPct;
      document.getElementById('xgbBar').style.width = xgbPct;
      document.getElementById('xgbBar').className = data.xgb_blocked ? 'bg-rose-600 h-full rounded-full transition-all duration-500' : 'bg-emerald-600 h-full rounded-full transition-all duration-500';
      const xgbBadge = document.getElementById('xgbBadge');
      xgbBadge.innerText = data.xgb_blocked ? 'FLAGGED' : 'CLEAR';
      xgbBadge.className = data.xgb_blocked ? 'px-1.5 py-0.5 rounded text-[10px] font-bold font-mono bg-rose-100 text-rose-800' : 'px-1.5 py-0.5 rounded text-[10px] font-bold font-mono bg-emerald-100 text-emerald-800';

      const lstmPct = (data.lstm_prob * 100).toFixed(1) + '%';
      document.getElementById('lstmScore').innerText = lstmPct;
      document.getElementById('lstmBar').style.width = lstmPct;
      document.getElementById('lstmBar').className = data.lstm_blocked ? 'bg-rose-600 h-full rounded-full transition-all duration-500' : 'bg-emerald-600 h-full rounded-full transition-all duration-500';
      const lstmBadge = document.getElementById('lstmBadge');
      lstmBadge.innerText = data.lstm_blocked ? 'FLAGGED' : 'CLEAR';
      lstmBadge.className = data.lstm_blocked ? 'px-1.5 py-0.5 rounded text-[10px] font-bold font-mono bg-rose-100 text-rose-800' : 'px-1.5 py-0.5 rounded text-[10px] font-bold font-mono bg-emerald-100 text-emerald-800';

      // Behavioral Factors
      const list = document.getElementById('factorsList');
      list.innerHTML = '';
      data.factors.forEach(f => {
        const div = document.createElement('div');
        div.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs ' + 
          (f.level === 'danger' ? 'bg-rose-50 text-rose-900 border-rose-200' : (f.level === 'safe' ? 'bg-emerald-50 text-emerald-900 border-emerald-200' : 'bg-[#faf9f3] text-neutral-800 border-[#e7e5dc]'));
        const badgeBg = f.level === 'danger' ? 'bg-rose-200 text-rose-900' : (f.level === 'safe' ? 'bg-emerald-200 text-emerald-900' : 'bg-neutral-200 text-neutral-800');
        div.innerHTML = `<span class="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded ${badgeBg}">${f.tag}</span><span>${f.desc}</span>`;
        list.appendChild(div);
      });
    }

    // Auto run default inspection on page load
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('txForm').requestSubmit();
    });
  </script>
</body>
</html>
"""


class FraudDetectionServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            amount = float(data.get("amount", 0.0))
            old_org = float(data.get("oldbalanceOrg", 0.0))
            new_org = float(data.get("newbalanceOrig", 0.0))
            old_dest = float(data.get("oldbalanceDest", 0.0))
            new_dest = float(data.get("newbalanceDest", 0.0))
            tx_type = data.get("type", "TRANSFER")
            tx_count = int(data.get("tx_count_last_N", 1))
            steps_since = float(data.get("steps_since_last_tx", 999.0))
            is_new_ben = int(data.get("is_new_beneficiary", 1))
            payer_id = data.get("payer_id", "default_payer")

            # Engineered features
            is_balance_drain = 1 if (new_org == 0.0 and old_org > 0.0) else 0
            type_transfer = 1 if tx_type == "TRANSFER" else 0
            type_cashout = 1 if tx_type == "CASH_OUT" else 0

            feature_dict = {
                "amount": amount,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
                "type_TRANSFER": type_transfer,
                "type_CASH_OUT": type_cashout,
                "tx_count_last_N": tx_count,
                "sum_amount_last_N": amount * tx_count,
                "distinct_dest_last_N": tx_count if is_new_ben else 1,
                "steps_since_last_tx": steps_since,
                "is_balance_drain": is_balance_drain,
                "is_new_beneficiary": is_new_ben,
            }

            raw_df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
            scaled_df = pd.DataFrame(scaler.transform(raw_df), columns=FEATURE_COLUMNS)

            # 1. XGBoost inference
            xgb_prob = float(xgb_model.predict_proba(scaled_df)[0])
            xgb_blocked = bool(xgb_prob >= xgb_model.threshold)

            # 2. Sequential LSTM inference
            lstm_prob, lstm_blocked_int = lstm_model.predict_step(scaled_df, payer_id)
            lstm_prob = float(lstm_prob)
            lstm_blocked = bool(lstm_blocked_int == 1)

            # Combined Gate Decision
            is_fraud = xgb_blocked or lstm_blocked

            # Dynamic Behavioral Attribution
            factors = []
            if is_balance_drain:
                factors.append({"tag": "DRAIN", "desc": "100% Account Balance Drain (Zero Balance remaining)", "level": "danger"})
            if is_new_ben:
                factors.append({"tag": "NOVEL DEST", "desc": "Beneficiary account has zero historical interactions", "level": "danger"})
            if amount >= 100000:
                factors.append({"tag": "HIGH VALUE", "desc": f"Large capital quantum transfer: ₹{amount:,.2f}", "level": "danger"})
            if type_transfer:
                factors.append({"tag": "TRANSFER", "desc": "Immediate outbound funds movement", "level": "neutral"})
            elif type_cashout:
                factors.append({"tag": "CASH OUT", "desc": "Immediate ATM/Cash liquidation attempt", "level": "danger"})
            if steps_since <= 1.0:
                factors.append({"tag": "VELOCITY", "desc": "High frequency burst: immediate rapid transaction delay", "level": "danger"})
            if not factors:
                factors.append({"tag": "ESTABLISHED", "desc": "Spending behavior matches normal historical account curve", "level": "safe"})

            response_data = {
                "is_fraud": is_fraud,
                "xgb_prob": xgb_prob,
                "xgb_blocked": xgb_blocked,
                "lstm_prob": lstm_prob,
                "lstm_blocked": lstm_blocked,
                "factors": factors,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def log_message(self, format, *args):
        return


def run_server():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, FraudDetectionServer)
    print("=" * 70)
    print(f"  🚀 FRAUDSHIELD AI WEBSITE IS LIVE AND FULLY CONNECTED!")
    print(f"  👉 Open in your browser: http://127.0.0.1:{PORT}/")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    run_server()
