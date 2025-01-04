// Add OpenAI specific test function at the beginning
function testOpenAIConnection() {
    fetch('/api/test/openai')
        .then(response => response.json())
        .then(data => {
            const openaiStatus = document.getElementById('openaiStatus');
            const openaiError = document.getElementById('openaiError');

            if (data.status === 'success') {
                openaiStatus.className = 'badge bg-success';
                openaiStatus.textContent = 'Connected';
                openaiError.style.display = 'none';
            } else {
                openaiStatus.className = 'badge bg-danger';
                openaiStatus.textContent = 'Error';
                if (data.message || data.details?.error) {
                    openaiError.textContent = data.message || data.details.error;
                    openaiError.style.display = 'inline';
                }
            }

            // Add detailed information to URL display
            const urlInput = document.getElementById('llmStatusUrl');
            if (urlInput) {
                urlInput.value = `${window.location.origin}/api/test/openai`;
            }
        })
        .catch(error => {
            console.error('Error testing OpenAI connection:', error);
            const openaiStatus = document.getElementById('openaiStatus');
            const openaiError = document.getElementById('openaiError');
            openaiStatus.className = 'badge bg-danger';
            openaiStatus.textContent = 'Error';
            openaiError.textContent = 'Failed to test connection';
            openaiError.style.display = 'inline';
        });
}

// Update the existing checkLLMStatus function to include the OpenAI specific test
function checkLLMStatus() {
    // Test OpenAI specifically first
    testOpenAIConnection();

    // Continue with general LLM status check
    fetch('/api/test/llm-status')
        .then(response => response.json())
        .then(status => {
            // Update Cohere status
            const cohereStatus = document.getElementById('cohereStatus');
            const cohereError = document.getElementById('cohereError');
            if (status.cohere.connected) {
                cohereStatus.className = 'badge bg-success';
                cohereStatus.textContent = 'Connected';
                cohereError.style.display = 'none';
            } else {
                cohereStatus.className = 'badge bg-danger';
                cohereStatus.textContent = 'Disconnected';
                if (status.cohere.error) {
                    cohereError.textContent = status.cohere.error;
                    cohereError.style.display = 'inline';
                }
            }

            // Update HuggingFace status
            const huggingfaceStatus = document.getElementById('huggingfaceStatus');
            const huggingfaceError = document.getElementById('huggingfaceError');
            if (status.huggingface && status.huggingface.connected) {
                huggingfaceStatus.className = 'badge bg-success';
                huggingfaceStatus.textContent = 'Connected';
                huggingfaceError.style.display = 'none';
            } else {
                huggingfaceStatus.className = 'badge bg-danger';
                huggingfaceStatus.textContent = 'Disconnected';
                if (status.huggingface && status.huggingface.error) {
                    huggingfaceError.textContent = status.huggingface.error;
                    huggingfaceError.style.display = 'inline';
                }
            }
        })
        .catch(error => {
            console.error('Error checking LLM status:', error);
            ['cohere', 'huggingface'].forEach(service => {
                const statusElement = document.getElementById(`${service}Status`);
                const errorElement = document.getElementById(`${service}Error`);
                statusElement.className = 'badge bg-danger';
                statusElement.textContent = 'Error';
                errorElement.textContent = 'Failed to check status';
                errorElement.style.display = 'inline';
            });
        });
}

// Initialize performance chart
const ctx = document.getElementById('performanceChart').getContext('2d');
const performanceChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Profit/Loss',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Update technical indicators
function updateTechnicalIndicators() {
    fetch('/api/technical_indicators')
        .then(response => response.json())
        .then(data => {
            // Update RSI & Stochastic
            if (data.rsi) {
                document.getElementById('rsiValue').textContent = data.rsi.value?.toFixed(2) || '-';
                document.getElementById('rsiSignal').textContent = data.rsi.signal || '-';
            }
            if (data.stochastic) {
                document.getElementById('stochK').textContent = data.stochastic.k_line?.toFixed(2) || '-';
                document.getElementById('stochD').textContent = data.stochastic.d_line?.toFixed(2) || '-';
                document.getElementById('stochSignal').textContent = data.stochastic.signal || '-';
            }

            // Update MACD
            if (data.macd) {
                document.getElementById('macdValue').textContent = data.macd.value?.toFixed(2) || '-';
                document.getElementById('macdSignal').textContent = data.macd.signal?.toFixed(2) || '-';
                document.getElementById('macdInterpretation').textContent = data.macd.interpretation || '-';
            }
        })
        .catch(error => {
            console.error('Error fetching technical indicators:', error);
            ['rsiValue', 'stochK', 'stochD', 'macdValue', 'macdSignal'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = '-';
            });
        });
}

// Update Telegram sentiment data
function updateTelegramSentiment(data) {
    if (data.telegram_sentiment) {
        document.getElementById('spydefiRating').textContent = data.telegram_sentiment.spydefi_rating || '-';
        document.getElementById('messageVolume').textContent = data.telegram_sentiment.message_count || '-';
        document.getElementById('sentimentScore').textContent = data.telegram_sentiment.sentiment_score?.toFixed(2) || '-';
        document.getElementById('positiveCount').textContent = data.telegram_sentiment.positive_count || '-';
        document.getElementById('negativeCount').textContent = data.telegram_sentiment.negative_count || '-';
        document.getElementById('neutralCount').textContent = data.telegram_sentiment.neutral_count || '-';

        // Format top keywords as a comma-separated list
        const keywords = data.telegram_sentiment.top_keywords;
        document.getElementById('topKeywords').textContent =
            Array.isArray(keywords) ? keywords.join(', ') : '-';
    }
}

// Update risk management data
function updateRiskManagement() {
    fetch('/api/risk_management')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            try {
                // Update portfolio metrics
                document.getElementById('openPositions').textContent = data.open_positions || '0';

                // Update chain allocations with Solana support
                const chains = ['bsc', 'eth', 'polygon', 'arb', 'avax', 'solana'];
                const totalAllocation = Object.values(data.chain_allocations || {}).reduce((sum, val) => sum + val, 0);

                chains.forEach(chain => {
                    const allocation = data.chain_allocations?.[chain] || 0;
                    const element = document.getElementById(`${chain}Allocation`);
                    if (element) {
                        // Calculate percentage relative to total allocation
                        const percentage = totalAllocation > 0 ? (allocation / totalAllocation) * 100 : 0;
                        element.style.width = `${percentage}%`;
                        element.textContent = `${chain.toUpperCase()}: ${percentage.toFixed(1)}%`;
                        element.setAttribute('aria-valuenow', percentage);
                    }
                });

                // Update risk parameters with safe access
                if (data.max_position_size !== undefined) {
                    const maxPosElement = document.getElementById('maxPositionSize');
                    if (maxPosElement) {
                        maxPosElement.textContent = `${(data.max_position_size * 100).toFixed(1)}%`;
                    }
                }
                if (data.stop_loss_percent !== undefined) {
                    const stopLossElement = document.getElementById('stopLossPercent');
                    if (stopLossElement) {
                        stopLossElement.textContent = `${(data.stop_loss_percent * 100).toFixed(1)}%`;
                    }
                }
                if (data.take_profit_percent !== undefined) {
                    const takeProfitElement = document.getElementById('takeProfitPercent');
                    if (takeProfitElement) {
                        takeProfitElement.textContent = `${(data.take_profit_percent * 100).toFixed(1)}%`;
                    }
                }

                // Update chain metrics with safe access
                const metricsTable = document.getElementById('chainMetrics');
                if (metricsTable && data.chain_metrics) {
                    let metricsHtml = '';
                    Object.entries(data.chain_metrics).forEach(([chain, metrics]) => {
                        if (metrics) {
                            metricsHtml += `
                                <tr>
                                    <td>${chain.toUpperCase()}</td>
                                    <td>${metrics.risk_score?.toFixed(2) || '-'}</td>
                                    <td>$${metrics.min_liquidity?.toLocaleString() || '-'}</td>
                                </tr>
                            `;
                        }
                    });
                    metricsTable.innerHTML = metricsHtml || '<tr><td colspan="3">No metrics available</td></tr>';
                }
            } catch (error) {
                console.error('Error updating risk management UI:', error);
            }
        })
        .catch(error => {
            console.error('Error fetching risk management data:', error);
            // Update UI to show error state
            const metricsTable = document.getElementById('chainMetrics');
            if (metricsTable) {
                metricsTable.innerHTML = '<tr><td colspan="3" class="text-danger">Failed to load risk management data</td></tr>';
            }
        });
}

// Update early launches data
function updateEarlyLaunches() {
    fetch('/api/early_launches')
        .then(response => response.json())
        .then(launches => {
            const launchesList = document.getElementById('earlyLaunchesList');
            if (!launches || !Array.isArray(launches)) {
                console.error('Invalid launches data received:', launches);
                launchesList.innerHTML = '<tr><td colspan="9" class="text-center">No early launches detected</td></tr>';
                return;
            }

            // Update potential counts
            const highPotential = launches.filter(l => l.potential_score >= 7).length;
            const mediumPotential = launches.filter(l => l.potential_score >= 5 && l.potential_score < 7).length;
            document.getElementById('highPotentialCount').textContent = `${highPotential} High Potential`;
            document.getElementById('mediumPotentialCount').textContent = `${mediumPotential} Medium Potential`;

            if (launches.length === 0) {
                launchesList.innerHTML = '<tr><td colspan="9" class="text-center">No early launches detected</td></tr>';
                return;
            }

            launchesList.innerHTML = launches.map(launch => {
                try {
                    const launchTime = new Date(launch.launch_time);
                    const potentialScore = launch.potential_score || 0;
                    const riskScore = launch.risk_score || 5;

                    const potentialScoreClass = potentialScore >= 7 ? 'text-success' :
                                              potentialScore >= 5 ? 'text-warning' : 'text-danger';
                    const riskClass = riskScore <= 3 ? 'text-success' :
                                        riskScore <= 6 ? 'text-warning' : 'text-danger';

                    return `
                        <tr>
                            <td><span class="text-truncate" style="max-width: 150px; display: inline-block;" title="${launch.token_address}">${launch.token_address}</span></td>
                            <td><span class="badge bg-${getChainBadgeColor(launch.chain)}">${launch.chain.toUpperCase()}</span></td>
                            <td>${launchTime.toLocaleString()}</td>
                            <td>$${(launch.liquidity || 0).toLocaleString()}</td>
                            <td>$${(launch.volume_24h || 0).toLocaleString()}</td>
                            <td>${(launch.holder_count || 0).toLocaleString()}</td>
                            <td>$${(launch.price || 0).toFixed(8)}</td>
                            <td class="${potentialScoreClass}">${potentialScore.toFixed(2)}</td>
                            <td class="${riskClass}">${riskScore.toFixed(2)}</td>
                        </tr>
                    `;
                } catch (err) {
                    console.error('Error processing launch:', launch, err);
                    return '';
                }
            }).join('');
        })
        .catch(error => {
            console.error('Error fetching early launches:', error);
            const launchesList = document.getElementById('earlyLaunchesList');
            launchesList.innerHTML = '<tr><td colspan="9" class="text-center text-danger">Error loading early launches</td></tr>';
        });
}

// Update trades section
function updateTrades() {
    fetch('/api/trades')
        .then(response => response.json())
        .then(trades => {
            const tradesList = document.getElementById('tradesList');
            if (!trades || !Array.isArray(trades)) {
                console.error('Invalid trades data received:', trades);
                tradesList.innerHTML = '<tr><td colspan="6" class="text-center">No trades available</td></tr>';
                document.getElementById('tradeCount').textContent = '0 trades';
                return;
            }

            // Update trade count
            document.getElementById('tradeCount').textContent = `${trades.length} trades`;

            if (trades.length === 0) {
                tradesList.innerHTML = '<tr><td colspan="6" class="text-center">No trades available</td></tr>';
                return;
            }

            // Take only unique trades based on timestamp + token combination
            const uniqueTrades = trades.reduce((acc, trade) => {
                const key = `${trade.timestamp}-${trade.token}-${trade.chain}`;
                if (!acc[key]) {
                    acc[key] = trade;
                }
                return acc;
            }, {});

            // Convert back to array and sort by timestamp
            const sortedTrades = Object.values(uniqueTrades)
                .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                .slice(0, 10); // Show last 10 trades

            tradesList.innerHTML = sortedTrades.map(trade => `
                <tr>
                    <td>${trade.token}</td>
                    <td><span class="badge bg-${getChainBadgeColor(trade.chain)}">${trade.chain}</span></td>
                    <td><span class="badge ${trade.type === 'BUY' ? 'bg-success' : 'bg-danger'}">${trade.type}</span></td>
                    <td>$${trade.price.toFixed(6)}</td>
                    <td>${trade.amount.toFixed(4)}</td>
                    <td>${new Date(trade.timestamp).toLocaleString()}</td>
                </tr>
            `).join('');
        })
        .catch(error => {
            console.error('Error fetching trades:', error);
            const tradesList = document.getElementById('tradesList');
            tradesList.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading trades</td></tr>';
            document.getElementById('tradeCount').textContent = '0 trades';
        });
}

// Helper function for chain badge colors
function getChainBadgeColor(chain) {
    const colors = {
        'BSC': 'warning',
        'ETH': 'primary',
        'POLYGON': 'info',
        'ARB': 'secondary',
        'AVAX': 'danger',
        'SOLANA': 'success'
    };
    return colors[chain] || 'secondary';
}

// Add chat functionality to existing dashboard.js

// Initialize chat
function initializeChat() {
    const chatForm = document.getElementById('chatForm');
    const userMessage = document.getElementById('userMessage');
    const chatMessages = document.getElementById('chatMessages');
    const advisorStatus = document.getElementById('advisorStatus');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userMessage.value.trim();
        if (!message) return;

        // Add user message to chat
        appendMessage('user', message);
        userMessage.value = '';

        // Update status
        advisorStatus.textContent = 'Thinking...';
        advisorStatus.className = 'badge bg-warning';

        try {
            const response = await fetch('/api/advisor/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            if (data.status === 'success') {
                appendMessage('advisor', data.response);
                advisorStatus.textContent = 'Ready';
                advisorStatus.className = 'badge bg-info';
            } else {
                throw new Error(data.message || 'Failed to get response');
            }
        } catch (error) {
            console.error('Chat error:', error);
            appendMessage('system', 'Sorry, I encountered an error. Please try again.');
            advisorStatus.textContent = 'Error';
            advisorStatus.className = 'badge bg-danger';
        }
    });
}

// Append a message to the chat
function appendMessage(type, content) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `${type}-message`;
    messageDiv.textContent = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// Add sentiment analysis functions to existing dashboard.js

function updateSentimentAnalysis() {
    fetch('/api/sentiment/analysis')
        .then(response => response.json())
        .then(data => {
            // Update Market Mood
            const moodValue = document.getElementById('marketMoodValue');
            const moodBadge = document.getElementById('marketMoodBadge');
            if (data.market_mood) {
                moodValue.textContent = `${data.market_mood.value}%`;
                moodBadge.textContent = data.market_mood.label;
                moodBadge.className = `badge ${getMoodClass(data.market_mood.value)}`;
            }

            // Update Social Sentiment
            const socialPositiveBar = document.getElementById('socialPositiveBar');
            const socialPositive = document.getElementById('socialPositive');
            if (data.social_sentiment) {
                socialPositiveBar.style.width = `${data.social_sentiment.positive_percentage}%`;
                socialPositive.textContent = data.social_sentiment.positive_mentions.toLocaleString();
            }

            // Update Volume Metrics
            const volumeBar = document.getElementById('volumeBar');
            const volumeChange = document.getElementById('volumeChange');
            if (data.volume_metrics) {
                volumeBar.style.width = `${data.volume_metrics.percentage}%`;
                volumeChange.textContent = data.volume_metrics.change_24h;
            }

            // Update Market Trends
            if (data.market_trends) {
                // Fear & Greed Index
                const fearGreedValue = document.getElementById('fearGreedValue');
                fearGreedValue.textContent = `${data.market_trends.fear_greed.label} (${data.market_trends.fear_greed.value})`;
                fearGreedValue.className = `badge ${getFearGreedClass(data.market_trends.fear_greed.value)}`;

                // Trend Strength
                const trendStrength = document.getElementById('trendStrength');
                trendStrength.textContent = `${data.market_trends.trend_strength.value} ${data.market_trends.trend_strength.direction}`;
                trendStrength.className = 'badge bg-info';

                // Market Momentum
                const marketMomentum = document.getElementById('marketMomentum');
                marketMomentum.textContent = `${data.market_trends.momentum.value} (${data.market_trends.momentum.signal})`;
                marketMomentum.className = `badge ${getMomentumClass(data.market_trends.momentum.value)}`;
            }
        })
        .catch(error => {
            console.error('Error fetching sentiment analysis:', error);
            // Reset values on error
            ['marketMoodValue', 'socialPositive', 'volumeChange', 'fearGreedValue', 'trendStrength', 'marketMomentum'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = '-';
            });
        });
}

// Helper functions for sentiment analysis
function getMoodClass(value) {
    if (value >= 75) return 'bg-success';
    if (value >= 50) return 'bg-info';
    if (value >= 25) return 'bg-warning';
    return 'bg-danger';
}

function getFearGreedClass(value) {
    if (value >= 75) return 'bg-danger';  // Extreme Greed
    if (value >= 50) return 'bg-warning'; // Greed
    if (value >= 25) return 'bg-info';    // Fear
    return 'bg-success';                  // Extreme Fear
}

function getMomentumClass(value) {
    switch (value.toLowerCase()) {
        case 'high': return 'bg-success';
        case 'medium': return 'bg-info';
        case 'low': return 'bg-warning';
        default: return 'bg-secondary';
    }
}

// Update dashboard data
function updateDashboard() {
    // Add Solana health check
    checkSolanaHealth();

    // Check LLM status
    checkLLMStatus();

    // Update all sections
    updateTrades();
    updateTechnicalIndicators();
    updateRiskManagement();
    updateEarlyLaunches();

    // Add sentiment analysis update
    updateSentimentAnalysis();

    // Update performance data
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            if (data.total_profit) {
                performanceChart.data.labels.push(new Date().toLocaleTimeString());
                performanceChart.data.datasets[0].data.push(data.total_profit);
                if (performanceChart.data.labels.length > 20) {
                    performanceChart.data.labels.shift();
                    performanceChart.data.datasets[0].data.shift();
                }
                performanceChart.update();

                document.getElementById('totalProfit').textContent = `$${data.total_profit.toFixed(2)}`;
                document.getElementById('buybackAmount').textContent = `$${data.buyback_allocated.toFixed(2)}`;
            }
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
            document.getElementById('totalProfit').textContent = '-';
            document.getElementById('buybackAmount').textContent = '-';
        });
    // Initialize chat if not already initialized
    if (!window.chatInitialized) {
        initializeChat();
        window.chatInitialized = true;
    }
}

// Add Solana health check function
function checkSolanaHealth() {
    fetch('/api/monitor/solana/health')
        .then(response => response.json())
        .then(data => {
            const statusBadge = document.getElementById('solanaStatus');
            const versionSpan = document.getElementById('solanaVersion');
            const blockhashSpan = document.getElementById('solanaBlockhash');
            const updateTime = document.getElementById('solanaUpdateTime');

            if (data.status === 'healthy') {
                statusBadge.className = 'badge bg-success';
                statusBadge.textContent = 'Healthy';
                versionSpan.textContent = data.version || '-';
                blockhashSpan.textContent = data.latest_blockhash || '-';
            } else {
                statusBadge.className = 'badge bg-danger';
                statusBadge.textContent = 'Error';
                versionSpan.textContent = '-';
                blockhashSpan.textContent = '-';
            }

            if (data.timestamp) {
                const timestamp = new Date(data.timestamp);
                updateTime.textContent = `Last updated: ${timestamp.toLocaleTimeString()}`;
            }
        })
        .catch(error => {
            console.error('Error checking Solana health:', error);
            const statusBadge = document.getElementById('solanaStatus');
            statusBadge.className = 'badge bg-danger';
            statusBadge.textContent = 'Error';
        });
}

// Update dashboard every 30 seconds
updateDashboard();
setInterval(updateDashboard, 30000);

// Add URL copy functionality at the end of the file

// Function to set up URL copy functionality
function setupUrlCopy() {
    const urlInput = document.getElementById('llmStatusUrl');
    const copyBtn = document.getElementById('copyUrlBtn');
    const confirmationMsg = document.getElementById('copyConfirmation');

    // Set the URL
    const baseUrl = window.location.origin;
    urlInput.value = `${baseUrl}/api/test/llm-status`;

    // Copy functionality
    copyBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(urlInput.value);
            // Show confirmation
            confirmationMsg.style.display = 'block';
            // Hide after 3 seconds
            setTimeout(() => {
                confirmationMsg.style.display = 'none';
            }, 3000);
        } catch (err) {
            console.error('Failed to copy URL:', err);
            // Fallback copy method
            urlInput.select();
            document.execCommand('copy');
            // Show confirmation
            confirmationMsg.style.display = 'block';
            setTimeout(() => {
                confirmationMsg.style.display = 'none';
            }, 3000);
        }
    });
}

// Initialize URL copy functionality
setupUrlCopy();