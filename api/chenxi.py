from http.server import BaseHTTPRequestHandler
import json
import random
import numpy as np
from datetime import datetime, timezone
from collections import Counter
import os
import math

# ==================== 至尊预测算法核心 ====================
class TrendMaster:
    def analyze_multidimensional_trend(self, data):
        if len(data) < 10:
            return self._get_base_trend_analysis()
        
        try:
            time_trend = self._analyze_time_series(data)
            value_trend = self._analyze_value_distribution(data)
            volatility_trend = self._analyze_volatility(data)
            
            data_sufficiency = min(1.0, len(data) / 30)
            fused_trend = self._fuse_trend_dimensions(
                time_trend, value_trend, volatility_trend, data_sufficiency
            )
            
            return {
                'type': 'multidimensional_trend',
                'strength': fused_trend['strength'],
                'direction': fused_trend['direction'],
                'confidence': fused_trend['confidence']
            }
        except Exception as e:
            return self._get_base_trend_analysis()
    
    def _analyze_time_series(self, data):
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        if len(sums) < 8:
            return {'strength': 0.4, 'direction': 'neutral'}
        
        window = min(5, len(sums) // 3)
        moving_avg = [np.mean(sums[i:i+window]) for i in range(len(sums)-window+1)]
        
        if len(moving_avg) < 2:
            return {'strength': 0.4, 'direction': 'neutral'}
        
        first_quarter = np.mean(moving_avg[:len(moving_avg)//4])
        last_quarter = np.mean(moving_avg[-len(moving_avg)//4:])
        
        if last_quarter > first_quarter + 2:
            direction = 'bullish'
            strength = min(0.9, (last_quarter - first_quarter) / 10)
        elif last_quarter < first_quarter - 2:
            direction = 'bearish' 
            strength = min(0.9, (first_quarter - last_quarter) / 10)
        else:
            direction = 'neutral'
            strength = 0.4
        
        return {
            'strength': strength,
            'direction': direction,
            'confidence': min(0.8, strength * 1.5)
        }
    
    def _analyze_value_distribution(self, data):
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        if len(sums) < 8:
            return {'strength': 0.4, 'bias': 'balanced'}
        
        avg_sum = np.mean(sums)
        
        if avg_sum > 15:
            bias = 'large'
            strength = min(0.8, (avg_sum - 14) / 7)
        elif avg_sum < 13:
            bias = 'small'
            strength = min(0.8, (14 - avg_sum) / 7)
        else:
            bias = 'balanced'
            strength = 0.4
        
        return {
            'strength': strength,
            'bias': bias,
            'average': avg_sum,
            'confidence': min(0.8, strength * 1.3)
        }
    
    def _analyze_volatility(self, data):
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        if len(sums) < 8:
            return {'volatility': 0.5, 'stability': 'medium'}
        
        changes = [abs(sums[i] - sums[i-1]) for i in range(1, len(sums))]
        avg_change = np.mean(changes) if changes else 0
        volatility = avg_change / 27
        
        if volatility < 0.15:
            stability = 'high'
        elif volatility < 0.3:
            stability = 'medium'
        else:
            stability = 'low'
        
        return {
            'volatility': volatility,
            'stability': stability,
            'avg_change': avg_change
        }
    
    def _fuse_trend_dimensions(self, time_trend, value_trend, volatility_trend, data_sufficiency):
        strengths = [
            time_trend['strength'],
            value_trend['strength'], 
            1.0 - volatility_trend['volatility']
        ]
        
        avg_strength = np.mean(strengths) * data_sufficiency
        
        if (time_trend['direction'] == 'bullish' and value_trend['bias'] == 'large'):
            direction = 'bullish'
            confidence_boost = 0.2
        elif (time_trend['direction'] == 'bearish' and value_trend['bias'] == 'small'):
            direction = 'bearish'
            confidence_boost = 0.2
        else:
            direction = 'neutral'
            confidence_boost = 0
        
        return {
            'strength': min(0.95, avg_strength),
            'direction': direction,
            'confidence': min(0.9, avg_strength + confidence_boost)
        }
    
    def _get_base_trend_analysis(self):
        return {
            'type': 'multidimensional_trend',
            'strength': 0.4,
            'direction': 'neutral',
            'confidence': 0.4
        }

class PatternGenius:
    def detect_intelligent_patterns(self, data):
        if len(data) < 15:
            return self._get_base_pattern_analysis()
        
        try:
            sequence_patterns = self._analyze_sequence_patterns(data)
            cluster_patterns = self._analyze_cluster_patterns(data)
            repetition_patterns = self._analyze_repetition_patterns(data)
            
            fused_patterns = self._fuse_pattern_analysis(
                sequence_patterns, cluster_patterns, repetition_patterns
            )
            
            return {
                'type': 'intelligent_patterns',
                'pattern_strength': fused_patterns['strength'],
                'pattern_type': fused_patterns['type'],
                'confidence': fused_patterns['confidence'],
                'next_expected': fused_patterns['next_expected']
            }
        except Exception as e:
            return self._get_base_pattern_analysis()

    def _analyze_sequence_patterns(self, data):
        results = [d['result'] for d in data if 'result' in d]
        if len(results) < 10:
            return {'strength': 0.4, 'pattern': 'random'}
        
        recent = results[:15]
        
        if len(set(recent[:5])) == 1:
            return {'strength': 0.8, 'pattern': 'stable', 'next_expected': recent[0]}
        
        if len(recent) >= 6:
            pattern_check = all(recent[i] != recent[i+1] for i in range(5))
            if pattern_check:
                expected = '大单' if recent[-1] == '大双' else '大双' if recent[-1] == '小单' else '小单' if recent[-1] == '小双' else '大单'
                return {'strength': 0.7, 'pattern': 'alternating', 'next_expected': expected}
        
        return {'strength': 0.5, 'pattern': 'random'}

    def _analyze_cluster_patterns(self, data):
        results = [d['result'] for d in data if 'result' in d]
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        
        if len(results) < 10:
            return {'strength': 0.4, 'clusters': []}
        
        result_counts = Counter(results)
        total = len(results)
        max_count = max(result_counts.values())
        cluster_strength = max_count / total if total > 0 else 0.4
        
        if len(sums) >= 10:
            sum_std = np.std(sums)
            sum_stability = 1.0 - (sum_std / 27)
            cluster_strength = (cluster_strength + sum_stability) / 2
        
        return {
            'strength': cluster_strength,
            'clusters': dict(result_counts),
            'most_common': result_counts.most_common(1)[0][0] if result_counts else None
        }

    def _analyze_repetition_patterns(self, data):
        results = [d['result'] for d in data if 'result' in d]
        if len(results) < 12:
            return {'strength': 0.4, 'repetition_rate': 0}
        
        repeats = 0
        for i in range(1, min(10, len(results))):
            if results[i] == results[i-1]:
                repeats += 1
        
        repetition_rate = repeats / min(10, len(results)-1)
        
        if repetition_rate > 0.6:
            strength = 0.8
        elif repetition_rate > 0.4:
            strength = 0.6
        else:
            strength = 0.4
        
        return {
            'strength': strength,
            'repetition_rate': repetition_rate,
            'next_expected': results[0] if repetition_rate > 0.5 else None
        }

    def _fuse_pattern_analysis(self, sequence, cluster, repetition):
        strengths = [sequence['strength'], cluster['strength'], repetition['strength']]
        avg_strength = np.mean(strengths)
        
        next_expected = None
        if sequence.get('next_expected'):
            next_expected = sequence['next_expected']
        elif repetition.get('next_expected'):
            next_expected = repetition['next_expected']
        elif cluster.get('most_common'):
            next_expected = cluster['most_common']
        
        confidence = min(0.9, avg_strength * 1.3)
        
        return {
            'strength': avg_strength,
            'type': 'strong' if avg_strength > 0.7 else 'moderate' if avg_strength > 0.5 else 'weak',
            'confidence': confidence,
            'next_expected': next_expected
        }

    def _get_base_pattern_analysis(self):
        return {
            'type': 'intelligent_patterns',
            'pattern_strength': 0.4,
            'pattern_type': 'random',
            'confidence': 0.4,
            'next_expected': None
        }

class RiskSage:
    def assess_comprehensive_risk(self, data):
        if len(data) < 20:
            return self._get_base_risk_assessment()
        
        try:
            volatility_risk = self._assess_volatility_risk(data)
            reversal_risk = self._assess_reversal_risk(data)
            consistency_risk = self._assess_consistency_risk(data)
            
            total_risk = self._fuse_risk_factors(volatility_risk, reversal_risk, consistency_risk)
            
            return {
                'type': 'comprehensive_risk',
                'total_risk': total_risk['score'],
                'risk_level': total_risk['level'],
                'recommendation': total_risk['recommendation']
            }
        except Exception as e:
            return self._get_base_risk_assessment()

    def _assess_volatility_risk(self, data):
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        if len(sums) < 10:
            return {'score': 0.5, 'level': 'medium'}
        
        changes = [abs(sums[i] - sums[i-1]) for i in range(1, len(sums))]
        avg_change = np.mean(changes) if changes else 0
        volatility = avg_change / 27
        
        if volatility > 0.35:
            score = 0.8
            level = 'high'
        elif volatility > 0.2:
            score = 0.5
            level = 'medium'
        else:
            score = 0.2
            level = 'low'
        
        return {'score': score, 'level': level}

    def _assess_reversal_risk(self, data):
        results = [d['result'] for d in data if 'result' in d]
        if len(results) < 10:
            return {'score': 0.5, 'level': 'medium'}
        
        current_streak = 1
        for i in range(1, min(8, len(results))):
            if results[i] == results[i-1]:
                current_streak += 1
            else:
                break
        
        if current_streak >= 5:
            score = 0.8
            level = 'high'
        elif current_streak >= 3:
            score = 0.6
            level = 'medium'
        else:
            score = 0.3
            level = 'low'
        
        return {'score': score, 'level': level}

    def _assess_consistency_risk(self, data):
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        if len(sums) < 10:
            return {'score': 0.5, 'level': 'medium'}
        
        sum_std = np.std(sums)
        consistency = 1.0 - (sum_std / 27)
        
        if consistency < 0.3:
            score = 0.8
            level = 'high'
        elif consistency < 0.6:
            score = 0.5
            level = 'medium'
        else:
            score = 0.2
            level = 'low'
        
        return {'score': 1.0 - consistency, 'level': level}

    def _fuse_risk_factors(self, volatility_risk, reversal_risk, consistency_risk):
        scores = [volatility_risk['score'], reversal_risk['score'], consistency_risk['score']]
        avg_score = np.mean(scores)
        
        if avg_score > 0.7:
            level = '高风险'
            recommendation = '波动较大，建议谨慎参考'
        elif avg_score > 0.4:
            level = '中风险'
            recommendation = '适度参考，控制投入'
        else:
            level = '低风险'
            recommendation = '数据稳定，可以重点参考'
        
        return {
            'score': avg_score,
            'level': level,
            'recommendation': recommendation
        }

    def _get_base_risk_assessment(self):
        return {
            'type': 'comprehensive_risk',
            'total_risk': 0.5,
            'risk_level': '中风险',
            'recommendation': '数据不足，建议谨慎参考'
        }

class ConfidenceOracle:
    def calculate_adaptive_confidence(self, data):
        if len(data) < 25:
            return self._get_base_confidence()
        
        try:
            data_quality_conf = self._calculate_data_quality_confidence(data)
            trend_consistency_conf = self._calculate_trend_consistency_confidence(data)
            pattern_stability_conf = self._calculate_pattern_stability_confidence(data)
            
            total_confidence = self._fuse_confidence_factors(
                data_quality_conf, trend_consistency_conf, pattern_stability_conf
            )
            
            return {
                'type': 'adaptive_confidence',
                'total_confidence': total_confidence['score'],
                'confidence_level': total_confidence['level']
            }
        except Exception as e:
            return self._get_base_confidence()

    def _calculate_data_quality_confidence(self, data):
        if len(data) < 10:
            return {'score': 0.4, 'level': 'low'}
        
        complete_count = sum(1 for d in data if all(k in d for k in ['period', 'numbers', 'number_sum', 'result']))
        completeness = complete_count / len(data)
        
        now = datetime.now(timezone.utc)
        time_diffs = [(now - d['timestamp']).total_seconds() for d in data if 'timestamp' in d]
        avg_time_diff = np.mean(time_diffs) if time_diffs else 3600
        timeliness = max(0, 1 - avg_time_diff / (6 * 3600))
        
        score = (completeness + timeliness) / 2
        
        if score > 0.8:
            level = 'high'
        elif score > 0.6:
            level = 'medium'
        else:
            level = 'low'
        
        return {'score': score, 'level': level}

    def _calculate_trend_consistency_confidence(self, data):
        sums = [d['number_sum'] for d in data if 'number_sum' in d]
        if len(sums) < 10:
            return {'score': 0.4, 'level': 'low'}
        
        changes = [sums[i] - sums[i-1] for i in range(1, len(sums))]
        positive_trends = sum(1 for c in changes if c > 0)
        negative_trends = sum(1 for c in changes if c < 0)
        
        consistency = 1 - abs(positive_trends - negative_trends) / len(changes) if changes else 0.5
        
        if consistency > 0.7:
            level = 'high'
        elif consistency > 0.5:
            level = 'medium'
        else:
            level = 'low'
        
        return {'score': consistency, 'level': level}

    def _calculate_pattern_stability_confidence(self, data):
        results = [d['result'] for d in data if 'result' in d]
        if len(results) < 10:
            return {'score': 0.4, 'level': 'low'}
        
        changes = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
        change_rate = changes / (len(results) - 1) if len(results) > 1 else 0.5
        stability = 1 - change_rate
        
        if stability > 0.7:
            level = 'high'
        elif stability > 0.5:
            level = 'medium'
        else:
            level = 'low'
        
        return {'score': stability, 'level': level}

    def _fuse_confidence_factors(self, data_quality, trend_consistency, pattern_stability):
        scores = [data_quality['score'], trend_consistency['score'], pattern_stability['score']]
        weights = [0.4, 0.35, 0.25]
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        if weighted_score > 0.75:
            level = '高'
        elif weighted_score > 0.55:
            level = '中'
        else:
            level = '低'
        
        return {
            'score': weighted_score,
            'level': level
        }

    def _get_base_confidence(self):
        return {
            'type': 'adaptive_confidence',
            'total_confidence': 0.4,
            'confidence_level': '低'
        }

class SupremePredictionEngine:
    def __init__(self):
        self.trend_master = TrendMaster()
        self.pattern_genius = PatternGenius() 
        self.risk_sage = RiskSage()
        self.confidence_oracle = ConfidenceOracle()
    
    def supreme_predict(self, data):
        try:
            # 使用固定随机种子确保结果一致
            random.seed(42)
            np.random.seed(42)
            
            analysis_results = [
                self.trend_master.analyze_multidimensional_trend(data),
                self.pattern_genius.detect_intelligent_patterns(data),
                self.risk_sage.assess_comprehensive_risk(data),
                self.confidence_oracle.calculate_adaptive_confidence(data)
            ]
            
            final_prediction = self._supreme_fusion(analysis_results, data)
            return final_prediction
            
        except Exception as e:
            return self._get_intelligent_fallback(data)
    
    def _supreme_fusion(self, results, data):
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return self._get_intelligent_fallback(data)
        
        weights = self._calculate_supreme_weights(valid_results, data)
        fused_probs = self._multidimensional_deep_fusion(valid_results, weights)
        return self._generate_supreme_prediction(fused_probs, valid_results, data, weights)
    
    def _calculate_supreme_weights(self, results, data):
        weights = []
        for result in results:
            result_type = result.get('type', 'unknown')
            
            if result_type == 'multidimensional_trend':
                base_weight = 0.40
            elif result_type == 'intelligent_patterns':
                base_weight = 0.35
            elif result_type == 'comprehensive_risk':
                base_weight = 0.15
            elif result_type == 'adaptive_confidence':
                base_weight = 0.10
            else:
                base_weight = 0.10
            
            quality_factor = min(1.0, len(data) / 25)
            confidence = result.get('confidence', 0.5)
            
            adjusted_weight = base_weight * quality_factor * (confidence + 0.3)
            weights.append(adjusted_weight)
        
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [0.40, 0.35, 0.15, 0.10]
        
        return weights
    
    def _multidimensional_deep_fusion(self, results, weights):
        result_types = ['大单', '大双', '小单', '小双']
        fused_probs = {rt: 0.0 for rt in result_types}
        
        for i, result in enumerate(results):
            weight = weights[i]
            
            if result['type'] == 'multidimensional_trend':
                probs = self._extract_probs_from_trend(result, result_types)
            elif result['type'] == 'intelligent_patterns':
                probs = self._extract_probs_from_patterns(result, result_types)
            elif result['type'] == 'comprehensive_risk':
                probs = self._extract_probs_from_risk(result, result_types)
            elif result['type'] == 'adaptive_confidence':
                probs = self._extract_probs_from_confidence(result, result_types)
            else:
                probs = {rt: 0.25 for rt in result_types}
            
            for rt in result_types:
                fused_probs[rt] += probs[rt] * weight
        
        total_prob = sum(fused_probs.values())
        if total_prob > 0:
            fused_probs = {k: v / total_prob for k, v in fused_probs.items()}
        else:
            fused_probs = {rt: 0.25 for rt in result_types}
        
        return fused_probs
    
    def _extract_probs_from_trend(self, trend_result, result_types):
        direction = trend_result.get('direction', 'neutral')
        strength = trend_result.get('strength', 0.4)
        confidence = trend_result.get('confidence', 0.4)
        
        if direction == 'bullish':
            base_probs = {'大单': 0.45, '大双': 0.35, '小单': 0.12, '小双': 0.08}
        elif direction == 'bearish':
            base_probs = {'大单': 0.08, '大双': 0.12, '小单': 0.35, '小双': 0.45}
        else:
            base_probs = {'大单': 0.28, '大双': 0.28, '小单': 0.22, '小双': 0.22}
        
        if strength > 0.6:
            for rt in result_types:
                if base_probs[rt] > 0.25:
                    base_probs[rt] += 0.15
                else:
                    base_probs[rt] = max(0.05, base_probs[rt] - 0.1)
        
        confidence_factor = confidence * 0.3
        for rt in result_types:
            if base_probs[rt] > 0.25:
                base_probs[rt] += confidence_factor
            else:
                base_probs[rt] = max(0.05, base_probs[rt] - confidence_factor)
        
        total = sum(base_probs.values())
        return {k: v/total for k, v in base_probs.items()}
    
    def _extract_probs_from_patterns(self, pattern_result, result_types):
        next_expected = pattern_result.get('next_expected', '')
        pattern_strength = pattern_result.get('pattern_strength', 0.4)
        confidence = pattern_result.get('confidence', 0.4)
        
        if next_expected in result_types and pattern_strength > 0.4:
            probs = {rt: 0.08 for rt in result_types}
            probs[next_expected] = 0.76
        elif next_expected in result_types:
            probs = {rt: 0.18 for rt in result_types}
            probs[next_expected] = 0.46
        else:
            if pattern_strength > 0.6:
                probs = {'大单': 0.32, '大双': 0.32, '小单': 0.18, '小双': 0.18}
            elif pattern_strength > 0.4:
                probs = {'大单': 0.30, '大双': 0.30, '小单': 0.20, '小双': 0.20}
            else:
                probs = {rt: 0.25 for rt in result_types}
        
        return probs
    
    def _extract_probs_from_risk(self, risk_result, result_types):
        risk_level = risk_result.get('risk_level', '中风险')
        
        if risk_level == '高风险':
            return {rt: 0.25 for rt in result_types}
        elif risk_level == '低风险':
            return {'大单': 0.32, '大双': 0.32, '小单': 0.18, '小双': 0.18}
        else:
            return {'大单': 0.28, '大双': 0.28, '小单': 0.22, '小双': 0.22}
    
    def _extract_probs_from_confidence(self, confidence_result, result_types):
        confidence_level = confidence_result.get('confidence_level', '低')
        
        if confidence_level == '高':
            return {'大单': 0.38, '大双': 0.38, '小单': 0.12, '小双': 0.12}
        elif confidence_level == '中':
            return {'大单': 0.32, '大双': 0.32, '小单': 0.18, '小双': 0.18}
        else:
            return {'大单': 0.28, '大双': 0.28, '小单': 0.22, '小双': 0.22}
    
    def _generate_supreme_prediction(self, fused_probs, results, data, weights):
        sorted_probs = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
        recommendation = self._generate_supreme_recommendation(sorted_probs, results)
        value_prediction = self._predict_supreme_value(data, fused_probs)
        overall_confidence = self._calculate_overall_supreme_confidence(results)
        
        return {
            'predictions': {
                rt: {
                    'probability': float(round(probs, 4)),
                    'percentage': f"{probs*100:.1f}%",
                    'confidence': self._get_confidence_level(probs)
                } for rt, probs in fused_probs.items()
            },
            'recommendation': recommendation,
            'value_prediction': value_prediction,
            'overall_confidence': overall_confidence,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fusion_method': 'supreme_neural_fusion_v2',
            'algorithm_version': '辰溪至尊版v3.0-服务器版'
        }
    
    def _generate_supreme_recommendation(self, sorted_probs, results):
        top_prob = sorted_probs[0][1] if sorted_probs else 0.25
        
        if top_prob > 0.35:
            focus_count = 1
            risk_level = '低风险'
            strategy = '🎯 单点突破策略'
        elif top_prob > 0.28:
            focus_count = 2  
            risk_level = '中风险'
            strategy = '🛡️ 双核防御策略'
        else:
            focus_count = 2
            risk_level = '中高风险'
            strategy = '🌊 重点分散策略'
        
        focus_list = [item[0] for item in sorted_probs[:focus_count]]
        exclude_item = sorted_probs[-1][0] if sorted_probs and len(sorted_probs) > 1 else ''
        
        return {
            'focus': focus_list,
            'exclude': exclude_item,
            'strategy': strategy,
            'risk_level': risk_level,
            'focus_count': focus_count,
            'top_probability': float(top_prob)
        }
    
    def _predict_supreme_value(self, data, fused_probs):
        if not data:
            return 14
        
        recent_sums = [d['number_sum'] for d in data[-8:] if 'number_sum' in d]
        if not recent_sums:
            return 14
        
        avg_sum = np.mean(recent_sums)
        large_prob = fused_probs.get('大单', 0) + fused_probs.get('大双', 0)
        small_prob = fused_probs.get('小单', 0) + fused_probs.get('小双', 0)
        
        if large_prob > 0.65:
            adjustment = random.randint(1, 3)
        elif large_prob > 0.5:
            adjustment = random.randint(0, 2)
        elif small_prob > 0.65:
            adjustment = random.randint(-3, -1)
        elif small_prob > 0.5:
            adjustment = random.randint(-2, 0)
        else:
            adjustment = random.randint(-1, 1)
        
        predicted_value = int(avg_sum + adjustment)
        return max(3, min(27, predicted_value))
    
    def _calculate_overall_supreme_confidence(self, results):
        if not results:
            return "中"
        
        confidences = []
        for result in results:
            if 'confidence' in result:
                conf = result['confidence']
                if isinstance(conf, (int, float)):
                    confidences.append(conf)
        
        if not confidences:
            return "中"
        
        avg_confidence = np.mean(confidences)
        
        if avg_confidence > 0.65:
            return "高"
        elif avg_confidence > 0.45:
            return "中"
        else:
            return "低"
    
    def _get_confidence_level(self, probability):
        if probability > 0.35:
            return "高"
        elif probability > 0.25:
            return "中"
        else:
            return "低"
    
    def _get_intelligent_fallback(self, data):
        result_types = ['大单', '大双', '小单', '小双']
        
        if data:
            recent_results = [d['result'] for d in data[-15:] if 'result' in d]
            if recent_results:
                result_counts = Counter(recent_results)
                total = len(recent_results)
                probs = {rt: result_counts.get(rt, 0) / total for rt in result_types}
                
                min_prob = 0.12
                max_prob = 0.38
                for rt in result_types:
                    probs[rt] = max(min_prob, min(max_prob, probs[rt]))
                
                total_prob = sum(probs.values())
                probs = {k: v/total_prob for k, v in probs.items()}
            else:
                probs = {'大单': 0.30, '大双': 0.30, '小单': 0.20, '小双': 0.20}
        else:
            probs = {'大单': 0.30, '大双': 0.30, '小单': 0.20, '小双': 0.20}
        
        top_prob = max(probs.values())
        focus_items = [rt for rt, p in probs.items() if p == top_prob][:2]
        
        return {
            'predictions': {
                rt: {
                    'probability': probs[rt],
                    'percentage': f"{probs[rt]*100:.1f}%",
                    'confidence': '中'
                } for rt in result_types
            },
            'recommendation': {
                'focus': focus_items,
                'exclude': '',
                'strategy': '智能回退策略',
                'risk_level': '中风险',
                'focus_count': len(focus_items),
                'top_probability': top_prob
            },
            'value_prediction': 14,
            'overall_confidence': '中',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fusion_method': 'intelligent_fallback_v2',
            'algorithm_version': '辰溪至尊版v3.0-服务器版'
        }

# ==================== HTTP处理器 ====================
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """处理GET请求"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            # 使用固定数据测试
            sample_data = self._get_sample_data()
            
            engine = SupremePredictionEngine()
            prediction = engine.supreme_predict(sample_data)
            
            response = {
                'status': 'success',
                'data': prediction,
                'message': '辰溪AI至尊版预测结果',
                'server_time': datetime.now(timezone.utc).isoformat(),
                'data_source': '服务器内置样本数据'
            }
            
        except Exception as e:
            response = {
                'status': 'error',
                'message': f'预测失败: {str(e)}',
                'server_time': datetime.now(timezone.utc).isoformat()
            }
        
        self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))
    
    def do_POST(self):
        """处理POST请求 - 接收客户端历史数据"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            client_data = json.loads(post_data.decode('utf-8'))
            
            historical_data = client_data.get('historical_data', [])
            
            # 验证数据格式
            if not self._validate_data(historical_data):
                raise ValueError("数据格式不正确")
            
            engine = SupremePredictionEngine()
            prediction = engine.supreme_predict(historical_data)
            
            response = {
                'status': 'success',
                'data': prediction,
                'message': '基于您提供的历史数据生成的预测',
                'server_time': datetime.now(timezone.utc).isoformat(),
                'data_source': '客户端提供数据',
                'data_count': len(historical_data)
            }
            
        except Exception as e:
            response = {
                'status': 'error',
                'message': f'预测失败: {str(e)}',
                'server_time': datetime.now(timezone.utc).isoformat()
            }
        
        self.wfile.write(json.dumps(response, ensure_ascii=False, indent=2).encode('utf-8'))
    
    def do_OPTIONS(self):
        """处理OPTIONS请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _get_sample_data(self):
        """获取样本数据"""
        return [
            {'period': '2024001', 'numbers': [1,2,3], 'number_sum': 6, 'result': '小双', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024002', 'numbers': [4,5,6], 'number_sum': 15, 'result': '大单', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024003', 'numbers': [2,3,4], 'number_sum': 9, 'result': '小单', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024004', 'numbers': [5,6,7], 'number_sum': 18, 'result': '大双', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024005', 'numbers': [3,4,5], 'number_sum': 12, 'result': '小双', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024006', 'numbers': [6,7,8], 'number_sum': 21, 'result': '大单', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024007', 'numbers': [1,3,5], 'number_sum': 9, 'result': '小单', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024008', 'numbers': [2,4,6], 'number_sum': 12, 'result': '小双', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024009', 'numbers': [3,5,7], 'number_sum': 15, 'result': '大单', 'timestamp': datetime.now(timezone.utc)},
            {'period': '2024010', 'numbers': [4,6,8], 'number_sum': 18, 'result': '大双', 'timestamp': datetime.now(timezone.utc)}
        ]
    
    def _validate_data(self, data):
        """验证数据格式"""
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, dict):
                return False
            if 'number_sum' not in item or 'result' not in item:
                return False
            if not (3 <= item['number_sum'] <= 27):
                return False
            if item['result'] not in ['大单', '大双', '小单', '小双']:
                return False
        
        return True
