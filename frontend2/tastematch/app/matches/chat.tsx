import React, { useState, useEffect, useCallback, useRef } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList, KeyboardAvoidingView, Platform, ActivityIndicator, Animated } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';
import api, { getProfile } from '../../src/services/api';

// WS URL: Replace http with ws and add endpoint
const getWsUrl = (apiUrl: string) => {
    return apiUrl.replace('http', 'ws') + '/chat/ws';
};

export default function ChatScreen() {
    const { id, username, initialMessage } = useLocalSearchParams();
    const router = useRouter();

    const [messages, setMessages] = useState<any[]>([]);
    const messagesRef = useRef<any[]>([]); // Ref to keep track of latest messages for callbacks
    const [inputText, setInputText] = useState('');
    const [loading, setLoading] = useState(true);
    const [myId, setMyId] = useState<number | null>(null);
    const [isTyping, setIsTyping] = useState(false); // Other user typing
    const [lastTypingTime, setLastTypingTime] = useState(0);

    const initialMessageSent = useRef(false);
    const ws = useRef<WebSocket | null>(null);
    const flatListRef = useRef<FlatList>(null);
    const pendingQueue = useRef<any[]>([]);
    const typingTimeoutRef = useRef<any>(null);
    const reconnectTimeout = useRef<any>(null);


    // Fade animation for typing indicator
    const fadeAnim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        if (isTyping) {
            Animated.timing(fadeAnim, { toValue: 1, duration: 300, useNativeDriver: true }).start();
        } else {
            Animated.timing(fadeAnim, { toValue: 0, duration: 300, useNativeDriver: true }).start();
        }
    }, [isTyping]);

    useEffect(() => {
        setupChat();
        loadPendingQueue();

        return () => {
            if (ws.current) {
                ws.current.onclose = null;
                ws.current.close();
            }
            if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
            if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
        };
    }, [id]);

    const loadPendingQueue = async () => {
        try {
            const savedQueue = await AsyncStorage.getItem(`pendingQueue_${id}`);
            if (savedQueue) {
                pendingQueue.current = JSON.parse(savedQueue);
                console.log("Loaded Pending Queue:", pendingQueue.current.length);
            }
        } catch (e) {
            console.error("Failed to load pending queue", e);
        }
    };

    const savePendingQueue = async () => {
        try {
            await AsyncStorage.setItem(`pendingQueue_${id}`, JSON.stringify(pendingQueue.current));
        } catch (e) {
            console.error("Failed to save pending queue", e);
        }
    };

    const setupChat = async () => {
        try {
            initialMessageSent.current = false;
            const profile = await getProfile();
            setMyId(profile.id);

            const history = await api.get(`/chat/history/${id}`);
            // Ensure status is handled if API returns it, else default to 'read' for old messages
            const formattedHistory = history.data.map((m: any) => ({ ...m, status: m.status || 'read' }));

            setMessages(prev => {
                // Merge: Keep persistent local pending messages that haven't been sent yet
                const currentPending = prev.filter(m => m.pending);
                const updated = [...formattedHistory, ...currentPending];
                messagesRef.current = updated; // Sync ref
                return updated;
            });

            setLoading(false);

            connectWebSocket(profile.id);
        } catch (e) {
            console.error("Setup Chat Failed", e);
            setLoading(false);
            // Retry setup if API fails
            if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
            reconnectTimeout.current = setTimeout(setupChat, 5000);
        }
    };

    const connectWebSocket = async (userId: number) => {
        const token = await SecureStore.getItemAsync('userToken');
        if (!token) return;

        const wsUrl = `${getWsUrl(api.defaults.baseURL || '')}/${userId}?token=${token}`;
        console.log("Connecting WS:", wsUrl);

        if (ws.current) ws.current.close();
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            console.log("WS Connected");
            handleInitialMessage(userId);
            processPendingQueue();

            // Mark all existing messages from them as read
            markMessagesAsRead();
        };

        ws.current.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                handleWebSocketMessage(msg, userId);
            } catch (err) {
                console.error("WS Parse Error", err);
            }
        };

        ws.current.onerror = (e) => console.log("WS Error", e);
    };

    const processPendingQueue = () => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            // Process queue carefully
            const queue = pendingQueue.current;
            let changed = false;
            while (queue.length > 0) {
                const payload = queue[0]; // Peek
                console.log("Sending Queued Message:", payload);
                try {
                    ws.current.send(JSON.stringify(payload));
                    queue.shift(); // Remove only if send succeeds
                    changed = true;
                } catch (e) {
                    console.error("Failed to send queued message", e);
                    break;
                }
            }
            if (changed) savePendingQueue();
        }
    };

    const handleInitialMessage = (userId: number) => {
        console.log("Checking Initial Message:", initialMessage, "Sent:", initialMessageSent.current);
        if (initialMessage && !initialMessageSent.current) {
            const content = Array.isArray(initialMessage) ? initialMessage[0] : initialMessage;
            console.log("Sending Initial Message:", content);
            sendMessage(content, userId);
            initialMessageSent.current = true;
        } else {
            console.log("No initial message to send or already sent.");
        }
    };

    const handleWebSocketMessage = (msg: any, userId: number) => {
        const otherId = Number(id);

        if (msg.type === 'typing') {
            if (msg.sender_id === otherId) {
                setIsTyping(true);
                // Auto-clear typing after 3 seconds of no updates
                if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
                typingTimeoutRef.current = setTimeout(() => setIsTyping(false), 3000);
            }
            return;
        }

        if (msg.type === 'status_update') {
            // Update status of our messages
            if (msg.message_ids) {
                const newStatus = msg.status; // 'delivered' or 'read'
                setMessages(prev => {
                    const updated = prev.map(m =>
                        msg.message_ids.includes(m.id) ? { ...m, status: newStatus } : m
                    );
                    messagesRef.current = updated;
                    return updated;
                });
            }
            return;
        }

        const isFromMe = msg.sender_id === userId;
        const isFromThem = msg.sender_id === otherId;

        if ((isFromMe && msg.receiver_id === otherId) || (isFromThem && msg.receiver_id === userId)) {
            setMessages(prev => {
                if (isFromMe) {
                    // Optimistic update confirmation
                    // Replace pending message with real one
                    const pendingIndex = prev.findIndex(m => m.pending && m.client_message_id === msg.client_message_id);
                    if (pendingIndex !== -1) {
                        const newMsgs = [...prev];
                        newMsgs[pendingIndex] = { ...msg, status: 'sent' }; // Server confirmed
                        messagesRef.current = newMsgs; // Sync ref
                        return newMsgs;
                    }
                }

                if (prev.some(m => m.id === msg.id)) return prev;

                // If it's a new message from THEM, mark it as read immediately if we are strictly watching
                // (In a real app, check if app is foregrounded)
                if (isFromThem) {
                    sendReadReceipt([msg.id]);
                }

                const updated = [...prev, msg];
                messagesRef.current = updated;
                return updated;
            });
            setTimeout(() => flatListRef.current?.scrollToEnd(), 100);
        }
    };

    const sendReadReceipt = (messageIds: number[]) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN && messageIds.length > 0) {
            ws.current.send(JSON.stringify({
                type: 'read',
                receiver_id: Number(id), // Send to the other user so they know
                message_ids: messageIds
            }));
        }
    };

    const markMessagesAsRead = () => {
        // Find all messages from them that are NOT read
        const unreadIds = messagesRef.current // Use Ref to get latest state in callbacks
            .filter(m => m.sender_id === Number(id) && m.status !== 'read')
            .map(m => m.id);

        if (unreadIds.length > 0) {
            sendReadReceipt(unreadIds);
        }
    };

    const sendMessage = (text: string = inputText, senderId: number = myId!) => {
        if (!text.trim() || !senderId) return;

        const content = text.trim();
        const clientMsgId = `msg-${Date.now()}-${Math.random()}`;
        const otherId = Number(id);

        const payload = {
            type: 'message',
            receiver_id: otherId,
            content: content,
            client_message_id: clientMsgId
        };

        const tempMsg = {
            id: Date.now(),
            sender_id: senderId,
            receiver_id: otherId,
            content: content,
            timestamp: new Date().toISOString(),
            pending: true,
            status: 'pending',
            client_message_id: clientMsgId
        };

        setMessages(prev => {
            const updated = [...prev, tempMsg];
            messagesRef.current = updated;
            return updated;
        });
        setInputText('');

        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(payload));
        } else {
            console.log("WS Not Open. Queued.");
            pendingQueue.current.push(payload);
            savePendingQueue();
        }
    };

    const handleInputTextChange = (text: string) => {
        setInputText(text);

        // Send typing indicator throttled (every 1.5s max)
        const now = Date.now();
        if (now - lastTypingTime > 1500 && text.length > 0) {
            if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                ws.current.send(JSON.stringify({
                    type: 'typing',
                    receiver_id: Number(id)
                }));
                setLastTypingTime(now);
            }
        }
    };

    const renderStatusIcon = (item: any) => {
        if (item.sender_id !== myId) return null;
        if (item.pending) return <Ionicons name="time-outline" size={14} color="#ccc" />;

        // Simplified Status:
        // Pending: Clock
        // Sent/Delivered: Single Grey Check
        // Read: Single Blue Check

        if (item.status === 'read') {
            return <Ionicons name="checkmark" size={16} color="#34B7F1" />;
        } else {
            // Sent or Delivered -> Single Grey
            return <Ionicons name="checkmark" size={16} color="#ccc" />;
        }
    };

    const renderItem = ({ item }: { item: any }) => {
        const isMe = item.sender_id === myId;
        return (
            <View style={[styles.bubbleContainer, isMe ? styles.rightContainer : styles.leftContainer]}>
                <View style={[styles.bubble, isMe ? styles.rightBubble : styles.leftBubble]}>
                    <Text style={[styles.msgText, isMe ? styles.rightText : styles.leftText]}>
                        {item.content}
                    </Text>
                    <View style={styles.metaContainer}>
                        <Text style={[styles.timeText, isMe ? styles.rightTime : styles.leftTime]}>
                            {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </Text>
                        {isMe && <View style={styles.statusContainer}>{renderStatusIcon(item)}</View>}
                    </View>
                </View>
            </View>
        );
    };

    if (loading) {
        return <View style={styles.center}><ActivityIndicator color="#FF3366" /></View>;
    }

    return (
        <KeyboardAvoidingView
            style={styles.container}
            behavior={Platform.OS === "ios" ? "padding" : undefined}
            keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
        >
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
                    <Ionicons name="arrow-back" size={24} color="#333" />
                </TouchableOpacity>
                <View style={styles.headerTitle}>
                    <Text style={styles.username}>{username || "Chat"}</Text>
                    <View style={styles.statusRow}>
                        {isTyping ? (
                            <Text style={styles.typingText}>typing...</Text>
                        ) : (
                            <View style={styles.onlineContainer}>
                                <View style={styles.onlineDot} />
                                <Text style={styles.onlineText}>Online</Text>
                            </View>
                        )}
                    </View>
                </View>
                <TouchableOpacity style={styles.menuBtn}>
                    <Ionicons name="ellipsis-vertical" size={24} color="#333" />
                </TouchableOpacity>
            </View>

            <FlatList
                ref={flatListRef}
                data={messages}
                renderItem={renderItem}
                keyExtractor={item => item.id.toString()}
                contentContainerStyle={styles.listContent}
                onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
            />

            <View style={styles.inputContainer}>
                <TextInput
                    style={styles.input}
                    placeholder="Type a message..."
                    value={inputText}
                    onChangeText={handleInputTextChange}
                    multiline
                />
                <TouchableOpacity style={styles.sendBtn} onPress={() => sendMessage()}>
                    <Ionicons name="send" size={20} color="#fff" />
                </TouchableOpacity>
            </View>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#e5ddd5', // WhatsApp-like background
    },
    center: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center'
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingTop: 50,
        paddingBottom: 10,
        paddingHorizontal: 15,
        backgroundColor: '#fff',
        borderBottomWidth: 1,
        borderBottomColor: '#ddd',
        elevation: 3,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
    },
    backBtn: {
        padding: 5
    },
    headerTitle: {
        flex: 1,
        marginLeft: 15,
    },
    username: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    statusRow: {
        height: 16,
        justifyContent: 'center',
    },
    typingText: {
        fontSize: 12,
        color: '#00BFA5', // Teal color for typing
        fontStyle: 'italic',
    },
    onlineContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    onlineDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: '#4CAF50',
        marginRight: 4,
    },
    onlineText: {
        fontSize: 12,
        color: '#666',
    },
    menuBtn: {
        padding: 5
    },
    listContent: {
        padding: 15,
        paddingBottom: 20
    },
    bubbleContainer: {
        marginBottom: 8,
        flexDirection: 'row',
        width: '100%',
    },
    leftContainer: {
        justifyContent: 'flex-start'
    },
    rightContainer: {
        justifyContent: 'flex-end'
    },
    bubble: {
        maxWidth: '80%',
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 8,
        minWidth: 80,
    },
    leftBubble: {
        backgroundColor: '#fff',
        borderTopLeftRadius: 0,
    },
    rightBubble: {
        backgroundColor: '#E7FFDB', // WhatsApp Green-ish
        borderTopRightRadius: 0,
    },
    msgText: {
        fontSize: 16,
        marginBottom: 4,
    },
    leftText: {
        color: '#333'
    },
    rightText: {
        color: '#333'
    },
    metaContainer: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        alignItems: 'center',
    },
    timeText: {
        fontSize: 11,
        marginRight: 4,
    },
    leftTime: {
        color: '#999'
    },
    rightTime: {
        color: '#999'
    },
    statusContainer: {
        marginLeft: 2
    },
    inputContainer: {
        flexDirection: 'row',
        padding: 10,
        alignItems: 'center',
        backgroundColor: '#fff',
        margin: 10,
        borderRadius: 25,
        elevation: 2,
    },
    input: {
        flex: 1,
        backgroundColor: '#fff',
        paddingHorizontal: 15,
        paddingVertical: 10,
        fontSize: 16,
        maxHeight: 100,
        color: '#333',
    },
    sendBtn: {
        backgroundColor: '#00BFA5',
        width: 40,
        height: 40,
        borderRadius: 20,
        justifyContent: 'center',
        alignItems: 'center',
        marginLeft: 10,
        elevation: 2
    }
});
