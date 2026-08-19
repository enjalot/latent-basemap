import { describe, it, expect } from "vitest";
import { wheelFactor, K_WHEEL, BUTTON_IN, BUTTON_OUT, DBLCLICK_IN } from "./zoom.js";

describe("wheelFactor (gentle, direction, deltaMode)", () => {
  it("one mouse notch (~100px) is ~1.12x — the tuned gentle step", () => {
    expect(wheelFactor(100, 0, 800)).toBeCloseTo(1.12, 2);
  });
  it("deltaY > 0 zooms OUT (factor > 1); deltaY < 0 zooms IN (factor < 1)", () => {
    expect(wheelFactor(50, 0, 800)).toBeGreaterThan(1);
    expect(wheelFactor(-50, 0, 800)).toBeLessThan(1);
  });
  it("is much gentler than the old fixed 1.18 step for a typical notch", () => {
    expect(wheelFactor(100, 0, 800)).toBeLessThan(1.18);
  });
  it("scales line-mode deltas up (~16px per line) so line wheels still move", () => {
    expect(wheelFactor(3, 1, 800)).toBeCloseTo(Math.exp(K_WHEEL * 48), 5);
  });
  it("clamps a momentum spike so one event can't teleport the view", () => {
    expect(wheelFactor(100000, 0, 800)).toBeLessThanOrEqual(2);
    expect(wheelFactor(-100000, 0, 800)).toBeGreaterThanOrEqual(0.5);
  });
});

describe("discrete zoom steps", () => {
  it("buttons are inverse in/out and moderate (~1.6x)", () => {
    expect(BUTTON_IN * BUTTON_OUT).toBeCloseTo(1, 6);
    expect(BUTTON_OUT).toBeCloseTo(1.6, 6);
    expect(BUTTON_IN).toBeLessThan(1);
  });
  it("double-click zooms in (factor < 1)", () => {
    expect(DBLCLICK_IN).toBeLessThan(1);
  });
});
