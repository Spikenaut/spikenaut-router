#[cfg(test)]
mod tests {
    use crate::{AhlRouter, DomainSignals, VerificationDomain};

    #[test]
    fn chemistry_text_activates_chemistry() {
        let mut router = AhlRouter::new();
        let d = router.route(
            "Balance the equation: NaOH + HCl → NaCl + H₂O. \
             Find the limiting reactant given 2.5 mol NaOH and 3.0 mol HCl. \
             Calculate the theoretical yield of NaCl.",
        );
        assert!(d.is_active(VerificationDomain::Chemistry));
        assert!(!d.is_active(VerificationDomain::DigitalLogic));
    }

    #[test]
    fn math_text_activates_mathematics() {
        let mut router = AhlRouter::new();
        let d = router.route(
            "Find the derivative of f(x) = sin(x²) at x = π/4. \
             Compute the definite integral from 0 to 1.",
        );
        assert!(d.is_active(VerificationDomain::Mathematics));
    }

    #[test]
    fn logic_text_activates_digital_logic() {
        let mut router = AhlRouter::new();
        let d = router.route(
            "Simplify F(A,B,C) = A'BC + AB'C + ABC' + ABC using a Karnaugh map. \
             Verify the FSM transition table for determinism.",
        );
        assert!(d.is_active(VerificationDomain::DigitalLogic));
    }

    #[test]
    fn empty_text_routes_nowhere() {
        let mut router = AhlRouter::new();
        let d = router.route("Hello, how are you today?");
        assert!(d.is_empty());
    }

    #[test]
    fn firing_rates_in_range() {
        let mut router = AhlRouter::new();
        let d = router.route("The stoichiometry of NaOH + HCl reaction involves moles.");
        for &rate in &d.firing_rates {
            assert!(
                rate >= 0.0 && rate <= 1.0,
                "firing rate out of range: {rate}"
            );
        }
    }

    #[test]
    fn positive_feedback_increases_weight() {
        let mut router = AhlRouter::new();
        let w_before = router.weight_matrix()[0][0];
        router.apply_feedback(VerificationDomain::Chemistry, 1.0);
        let w_after = router.weight_matrix()[0][0];
        assert!(w_after > w_before);
    }

    #[test]
    fn negative_feedback_decreases_weight() {
        let mut router = AhlRouter::new();
        let w_before = router.weight_matrix()[0][0];
        router.apply_feedback(VerificationDomain::Chemistry, -1.0);
        let w_after = router.weight_matrix()[0][0];
        assert!(w_after < w_before);
    }

    #[test]
    fn domain_signals_extraction_chemistry_dominant() {
        let s = DomainSignals::from_text(
            "The molarity of the NaOH solution is 0.5 M. \
             Calculate the moles needed for the stoichiometry.",
        );
        assert!(s.chemistry > 0.0);
        assert!(s.chemistry > s.mathematics);
        assert!(s.chemistry > s.digital_logic);
    }

    #[test]
    fn domain_signals_empty_text_all_zero() {
        let s = DomainSignals::from_text("The sky is blue.");
        assert_eq!(s.chemistry, 0.0);
        assert_eq!(s.mathematics, 0.0);
        assert_eq!(s.digital_logic, 0.0);
    }

    #[test]
    fn total_routes_increments() {
        let mut router = AhlRouter::new();
        assert_eq!(router.total_routes, 0);
        router.route("hello");
        router.route("world");
        assert_eq!(router.total_routes, 2);
    }

    #[test]
    fn verification_domain_index_roundtrip() {
        for d in VerificationDomain::ALL {
            let idx = d.index();
            assert_eq!(VerificationDomain::from_index(idx), Some(d));
        }
        assert_eq!(VerificationDomain::from_index(99), None);
    }

    #[test]
    fn lif_neuron_fires_above_threshold() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        n.threshold = 0.1;
        n.decay_rate = 0.0; // no leak for this test
        n.integrate(0.5);
        assert!(n.check_fire().is_some());
        assert_eq!(n.membrane_potential, 0.0); // hard reset
    }

    #[test]
    fn lif_neuron_no_fire_below_threshold() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        n.threshold = 1.0;
        n.integrate(0.05);
        assert!(n.check_fire().is_none());
        assert!(n.membrane_potential > 0.0);
    }

    #[test]
    fn lif_neuron_adaptation_increases_on_fire() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        n.threshold = 0.1;
        n.decay_rate = 0.0;
        assert_eq!(n.adaptation, 0.0);
        n.integrate(0.5);
        n.check_fire();
        assert!(n.adaptation > 0.0);
        assert!(n.total_spikes > 0);
    }

    #[test]
    fn lif_neuron_adaptation_decays_over_time() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        n.adaptation = 0.5;
        n.adaptation_decay = 0.1;
        for _ in 0..10 {
            n.integrate(0.0);
        }
        assert!(n.adaptation < 0.5);
    }

    #[test]
    fn sparse_synaptic_map_from_dense() {
        use crate::sparse::SparseSynapticMap;
        const N: usize = 3;
        let matrix: [[f32; N]; N] = [
            [0.9, -0.15, -0.15],
            [-0.15, 0.9, -0.15],
            [-0.15, -0.15, 0.9],
        ];
        let sparse = SparseSynapticMap::<N>::from_dense(&matrix, 0.1);
        assert_eq!(sparse.nnz(), 9);
        assert!((sparse.sparsity() - 0.0).abs() < 0.001);
    }

    #[test]
    fn sparse_synaptic_map_prunes_small_weights() {
        use crate::sparse::SparseSynapticMap;
        const N: usize = 3;
        let matrix: [[f32; N]; N] = [
            [0.9, 0.001, 0.001],
            [0.001, 0.9, 0.001],
            [0.001, 0.001, 0.9],
        ];
        let sparse = SparseSynapticMap::<N>::from_dense(&matrix, 0.01);
        assert_eq!(sparse.nnz(), 3);
        assert!(sparse.sparsity() > 0.5);
    }

    #[test]
    fn sparse_synaptic_map_roundtrip() {
        use crate::sparse::SparseSynapticMap;
        const N: usize = 3;
        let matrix: [[f32; N]; N] = [
            [0.9, -0.15, -0.15],
            [-0.15, 0.9, -0.15],
            [-0.15, -0.15, 0.9],
        ];
        let sparse = SparseSynapticMap::<N>::from_dense(&matrix, 0.01);
        let recovered = sparse.to_dense();
        for i in 0..N {
            for j in 0..N {
                assert!((matrix[i][j] - recovered[i][j]).abs() < 0.001);
            }
        }
    }

    #[test]
    fn sparse_synaptic_map_builder() {
        use crate::sparse::SparseSynapticMapBuilder;
        const N: usize = 3;
        let map = SparseSynapticMapBuilder::<N>::new()
            .with_self_weight(0.9)
            .with_self_connections()
            .with_lateral_inhibition(-0.15)
            .build();
        assert_eq!(map.nnz(), 9);
        assert!((map.get_weight(0, 0) - 0.9).abs() < 0.001);
        assert!((map.get_weight(0, 1) - (-0.15)).abs() < 0.001);
    }

    #[test]
    fn telemetry_snapshot_creation() {
        use crate::sparse::TelemetrySnapshot;
        let snap = TelemetrySnapshot::new(3);
        assert_eq!(snap.adaptation.len(), 3);
        assert_eq!(snap.spike_counts.len(), 3);
        assert_eq!(snap.quant_error.len(), 3);
        assert_eq!(snap.step, 0);
    }

    #[test]
    fn telemetry_adaptation_penalty() {
        use crate::sparse::TelemetrySnapshot;
        let mut snap = TelemetrySnapshot::new(3);
        snap.adaptation[0] = 0.8;
        snap.adaptation[1] = 0.1;
        assert!(snap.adaptation_penalty(0, 1.0) > snap.adaptation_penalty(1, 1.0));
    }

    #[test]
    fn telemetry_quant_bonus() {
        use crate::sparse::TelemetrySnapshot;
        let mut snap = TelemetrySnapshot::new(3);
        snap.quant_error[0] = 0.9;
        snap.quant_error[1] = 0.1;
        assert!(snap.quant_bonus(1, 1.0) > snap.quant_bonus(0, 1.0));
    }

    #[test]
    fn routing_policy_scoring() {
        use crate::sparse::{RoutingPolicy, TelemetrySnapshot};
        let mut snap = TelemetrySnapshot::new(3);
        snap.spike_counts[0] = 10;
        snap.adaptation[0] = 0.1;
        snap.quant_error[0] = 0.1;
        snap.spike_counts[1] = 2;
        snap.adaptation[1] = 0.9;
        snap.quant_error[1] = 0.8;

        let policy = RoutingPolicy::default();
        let score0 = policy.score(0, &snap, 0.9);
        let score1 = policy.score(1, &snap, 0.9);
        assert!(score0 > score1);
    }

    #[test]
    fn router_to_sparse_conversion() {
        let router = AhlRouter::new();
        let sparse = router.to_sparse_map(0.01);
        assert!(sparse.nnz() > 0);
        assert!((sparse.get_weight(0, 0) - 0.9).abs() < 0.001);
    }

    #[test]
    fn router_capture_telemetry() {
        let mut router = AhlRouter::new();
        router.route("Balance NaOH + HCl");
        let telemetry = router.capture_telemetry(42);
        assert_eq!(telemetry.step, 42);
        assert_eq!(telemetry.adaptation.len(), 3);
    }

    #[test]
    fn router_adaptive_routing() {
        let mut router = AhlRouter::new();
        let mut telemetry = crate::sparse::TelemetrySnapshot::new(3);
        telemetry.adaptation[0] = 0.9;
        telemetry.adaptation[1] = 0.0;
        telemetry.adaptation[2] = 0.0;
        let decision = router.route_adaptive("Balance NaOH + HCl", &telemetry);
        assert!(decision.is_active(VerificationDomain::Chemistry));
    }

    #[test]
    fn router_policy_routing() {
        let mut router = AhlRouter::new();
        let telemetry = crate::sparse::TelemetrySnapshot::new(3);
        let policy = crate::sparse::RoutingPolicy::default();
        let decision = router.route_with_policy("Balance NaOH + HCl", &telemetry, &policy);
        assert!(decision.is_active(VerificationDomain::Chemistry));
    }

    #[test]
    fn router_telemetry_csv_output() {
        let router = AhlRouter::new();
        let telemetry = router.capture_telemetry(1);
        let firing_rates = [0.5, 0.0, 0.0];
        let csv = router.telemetry_csv(&telemetry, &firing_rates);
        assert!(csv.contains("step,domain,adaptation,spike_count,quant_error,firing_rate,active"));
        assert!(csv.contains("Deep Chemistry"));
        assert!(csv.contains("Advanced Math"));
        assert!(csv.contains("Digital Logic"));
    }

    #[test]
    fn lif_quant_error_estimate() {
        use crate::LifNeuron;
        let mut n = LifNeuron::new();
        assert!((n.quant_error_estimate() - 0.05).abs() < 0.001);
        n.adaptation = 1.0;
        assert!((n.quant_error_estimate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn sparse_map_gpu_arrays() {
        use crate::sparse::SparseSynapticMapBuilder;
        const N: usize = 3;
        let map = SparseSynapticMapBuilder::<N>::new()
            .with_self_weight(0.9)
            .with_self_connections()
            .build();
        let (row_ptr, col_idx, values) = map.to_gpu_arrays();
        assert_eq!(row_ptr.len(), N + 1);
        assert_eq!(col_idx.len(), N);
        assert_eq!(values.len(), N);
    }
}
