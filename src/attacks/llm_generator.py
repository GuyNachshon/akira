from collections import defaultdict
from typing import Dict, List, Any, Optional, Set

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.environment.state import NodeState


class LLMAttackGenerator:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.attack_history = []
        self.success_patterns = defaultdict(float)
        self.pattern_weights = {
            'scan_vulnerable': 0.7,
            'phishing_user': 0.6,
            'exploit_vulnerable': 0.8,
            'privilege_escalation_compromised': 0.5,
            'lateral_movement_compromised': 0.6,
            'data_exfiltration_compromised': 0.7,
            'persistence_compromised': 0.5,
            'defense_evasion_active': 0.6,
            'scan_high_value': 0.8,
            'exploit_isolated': 0.4,
            'lateral_movement_high_value': 0.7,
            'exploit_critical_path': 0.8
        }

    def _create_attack_prompt(self, network_state: Dict) -> str:
        """Create a prompt for attack generation based on network state."""
        # Analyze network state
        high_value_targets = self._identify_high_value_targets(network_state)
        vulnerable_nodes = sum(1 for node in network_state.values() if node.vulnerability > 7)
        isolated_nodes = sum(1 for node in network_state.values() if node.state == NodeState.ISOLATED)

        # Build prompt
        prompt = f"""Network analysis:
        - {vulnerable_nodes} highly vulnerable nodes detected
        - {len(high_value_targets)} high-value targets identified
        - {isolated_nodes} nodes currently isolated
        
        Generate a sophisticated attack sequence considering:
        1. Initial access points
        2. Lateral movement paths
        3. Target prioritization
        4. Evasion techniques
        
        Attack sequence should follow the pattern:
        <step>action|target_type|probability</step>
        
        Example:
        <step>phishing|vulnerable|0.8</step>
        <step>privilege_escalation|compromised|0.7</step>
"""

        return prompt

    def generate_attack_sequence(
            self,
            network_state: Dict,
            previous_success: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Generate attack sequence with adaptive learning."""
        # Update success patterns if feedback available
        if previous_success is not None and self.attack_history:
            last_pattern = self.attack_history[-1]
            self.success_patterns[last_pattern] = (
                    0.9 * self.success_patterns[last_pattern] +
                    0.1 * previous_success
            )

        # Analyze network state
        network_analysis = self._analyze_network(network_state)

        # Generate base prompt
        base_prompt = self._create_attack_prompt(network_state)

        # Add successful patterns to prompt
        successful_patterns = self._get_successful_patterns()
        if successful_patterns:
            base_prompt += "\nPreviously successful patterns:\n"
            base_prompt += "\n".join(successful_patterns)

        # Generate and refine attack sequence
        attack_sequence = self._generate_refined_sequence(
            base_prompt, network_analysis
        )

        # Store generated pattern
        pattern_hash = hash(str(attack_sequence))
        self.attack_history.append(pattern_hash)

        return attack_sequence

    def _identify_high_value_targets(self, network_state: Dict) -> Set[str]:
        """Identify high-value targets in the network."""
        high_value_targets = set()

        for node_id, node in network_state.items():
            # Consider a node high-value if it meets any of these criteria
            if (node.value > 7 or  # High business value
                    node.vulnerability > 8 or  # Highly vulnerable
                    len(list(network_state.get('neighbors', {}).get(node_id, []))) > 5):  # High connectivity
                high_value_targets.add(node_id)

        return high_value_targets

    def _parse_attack_steps(self, generated_text: str) -> List[Dict[str, Any]]:
        """Parse generated text into structured attack steps."""
        steps = []

        # Split into lines and process each step
        for line in generated_text.split('\n'):
            if '<step>' in line and '</step>' in line:
                # Extract step content
                step_content = line.split('<step>')[1].split('</step>')[0]

                try:
                    # Parse the step components
                    action, target_type, probability = step_content.split('|')

                    steps.append({
                        'action': action.strip(),
                        'target_type': target_type.strip(),
                        'probability': float(probability),
                        'description': self._generate_step_description(action.strip())
                    })
                except (ValueError, IndexError):
                    continue  # Skip malformed steps

        # If no valid steps were parsed, generate a default step
        if not steps:
            steps.append({
                'action': 'scan',
                'target_type': 'vulnerable',
                'probability': 0.7,
                'description': 'Perform initial network scan'
            })

        return steps

    def _analyze_network(self, network_state: Dict) -> Dict[str, Any]:
        """Perform detailed network analysis."""
        analysis = {
            'vulnerable_clusters': self._find_vulnerable_clusters(network_state),
            'critical_paths': self._identify_critical_paths(network_state),
            'high_value_targets': self._identify_high_value_targets(network_state)
        }
        return analysis

    def _find_vulnerable_clusters(self, network_state: Dict) -> List[Set[str]]:
        """Find clusters of vulnerable nodes."""
        vulnerable_nodes = {
            node_id for node_id, node in network_state.items()
            if node.vulnerability > 7
        }
        # Use networkx to find connected components
        return list(vulnerable_nodes)

    def _identify_critical_paths(self, network_state: Dict) -> List[List[str]]:
        """Identify critical paths through the network."""
        # Implementation would use networkx for path analysis
        return []

    def _get_successful_patterns(self) -> List[str]:
        """Get most successful attack patterns."""
        return [
            pattern for pattern, success in sorted(
                self.success_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        ]

    def _generate_refined_sequence(
            self,
            base_prompt: str,
            network_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate and refine attack sequence based on network analysis."""
        # Generate initial sequence
        inputs = self.tokenizer.encode(base_prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(outputs[0])

        # Parse into initial steps
        initial_steps = self._parse_attack_steps(generated_text)

        # Refine based on network analysis
        refined_steps = []
        for step in initial_steps:
            refined_step = step.copy()

            # Adjust probabilities based on vulnerable clusters
            if network_analysis['vulnerable_clusters']:
                refined_step['probability'] *= 1.2

            # Adjust target selection based on critical paths
            if network_analysis['critical_paths']:
                refined_step['target_type'] = 'critical_path'

            # Prioritize high-value targets
            if network_analysis['high_value_targets']:
                refined_step['target_type'] = 'high_value'
                refined_step['probability'] *= 1.1

            refined_steps.append(refined_step)

        return refined_steps

    def _generate_step_description(self, action: str) -> str:
        """Generate a detailed description for an attack step."""
        descriptions = {
            'scan': 'Perform reconnaissance to identify vulnerable targets',
            'phishing': 'Launch targeted phishing attacks against user accounts',
            'exploit': 'Exploit known vulnerabilities in target systems',
            'privilege_escalation': 'Attempt to gain elevated privileges on compromised systems',
            'lateral_movement': 'Move laterally through the network to reach additional targets',
            'data_exfiltration': 'Extract sensitive data from compromised systems',
            'persistence': 'Establish persistent access mechanisms',
            'defense_evasion': 'Deploy techniques to evade detection systems'
        }

        return descriptions.get(action, f'Execute {action} attack step')

    def _evaluate_step_success(self, step: Dict[str, Any], network_state: Dict) -> float:
        """Evaluate the potential success rate of an attack step."""
        base_probability = step['probability']

        # Adjust based on network state
        if step['target_type'] == 'vulnerable':
            vulnerable_nodes = sum(1 for node in network_state.values() if node.vulnerability > 7)
            if vulnerable_nodes > len(network_state) * 0.3:  # If more than 30% nodes are vulnerable
                base_probability *= 1.2

        # Adjust based on defense state
        isolated_nodes = sum(1 for node in network_state.values() if node.state == NodeState.ISOLATED)
        if isolated_nodes > len(network_state) * 0.2:  # If more than 20% nodes are isolated
            base_probability *= 0.8

        return min(1.0, base_probability)

    def _update_success_patterns(self, success_rate: float):
        """Update internal patterns based on attack success."""
        if not self.attack_history:
            return

        last_sequence = self.attack_history[-1]

        # Update pattern weights based on success
        for step in last_sequence:
            key = f"{step['action']}_{step['target_type']}"
            self.pattern_weights[key] = (
                    0.9 * self.pattern_weights.get(key, 0.5) +
                    0.1 * success_rate
            )
