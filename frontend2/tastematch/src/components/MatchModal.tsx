import React, { useEffect, useState } from 'react';
import { Modal, View, Text, StyleSheet, Image, TouchableOpacity, ActivityIndicator, Dimensions, Animated, TextInput, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

const { width, height } = Dimensions.get('window');

interface MatchModalProps {
    visible: boolean;
    onClose: () => void;
    matchedUser: any | null;
    currentUserImage?: string; // Optional: Show side-by-side
}

export default function MatchModal({ visible, onClose, matchedUser }: MatchModalProps) {
    const [loading, setLoading] = useState(true);
    const [opacity] = useState(new Animated.Value(0));
    const [scale] = useState(new Animated.Value(0.5));
    const router = useRouter();
    const [message, setMessage] = useState('');

    useEffect(() => {
        if (visible) {
            setLoading(true);
            // Suspense simulation (1.5 seconds)
            setTimeout(() => {
                setLoading(false);
                startAnimation();
            }, 1500);
        }
    }, [visible]);

    const startAnimation = () => {
        Animated.parallel([
            Animated.timing(opacity, {
                toValue: 1,
                duration: 500,
                useNativeDriver: true,
            }),
            Animated.spring(scale, {
                toValue: 1,
                friction: 5,
                useNativeDriver: true,
            })
        ]).start();
    };

    const handleSendMessage = () => {
        if (!message.trim()) return;

        console.log("Sending message: ", message);
        onClose();
        router.push({
            pathname: '/matches/chat',
            params: {
                id: matchedUser.user_id,
                username: matchedUser.username,
                initialMessage: message
            }
        });
    };

    if (!visible) return null;

    return (
        <Modal transparent animationType="fade" visible={visible}>
            <View style={styles.overlay}>
                {loading ? (
                    <View style={styles.loadingContainer}>
                        <View style={styles.radarRing}>
                            <ActivityIndicator size="large" color="#fff" />
                        </View>
                        <Text style={styles.loadingText}>Finding Match...</Text>
                    </View>
                ) : (
                    <View style={styles.content}>
                        <Animated.View style={[styles.successContent, { opacity, transform: [{ scale }] }]}>
                            <Text style={styles.matchTitle}>IT'S A MATCH!</Text>

                            <View style={styles.avatars}>
                                {/* Placeholder for Current User (Left) */}
                                <View style={[styles.avatarFrame, styles.leftAvatar]}>
                                    <Ionicons name="person" size={50} color="#ccc" />
                                </View>

                                {/* Matched User (Right) */}
                                <View style={[styles.avatarFrame, styles.rightAvatar]}>
                                    {matchedUser?.profile_image ? (
                                        <Image source={{ uri: matchedUser.profile_image }} style={styles.avatarImage} />
                                    ) : (
                                        <View style={styles.placeholderAvatar}>
                                            <Text style={styles.initial}>{matchedUser?.username?.[0]?.toUpperCase()}</Text>
                                        </View>
                                    )}
                                </View>

                                <View style={styles.heartIcon}>
                                    <Ionicons name="heart" size={40} color="#fff" />
                                </View>
                            </View>

                            <Text style={styles.matchName}>You and {matchedUser?.username} liked each other!</Text>

                            {matchedUser?.similarity && (
                                <View style={styles.scoreBadge}>
                                    <Text style={styles.scoreText}>{(matchedUser.similarity * 100).toFixed(0)}% Match</Text>
                                </View>
                            )}

                            {/* Tags */}
                            <View style={styles.tagsContainer}>
                                {matchedUser?.tags?.map((tag: string, index: number) => (
                                    <View key={index} style={styles.tagBadge}>
                                        <Text style={styles.tagText}>{tag}</Text>
                                    </View>
                                ))}
                            </View>

                            {/* Bio */}
                            {matchedUser?.bio && (
                                <Text style={styles.bioText} numberOfLines={3}>
                                    "{matchedUser.bio}"
                                </Text>
                            )}

                            <Text style={styles.icebreaker}>
                                {matchedUser?.match_reason || "Start the conversation now!"}
                            </Text>

                            <TextInput
                                style={styles.input}
                                placeholder="Say hello..."
                                placeholderTextColor="#ccc"
                                value={message}
                                onChangeText={setMessage}
                            />

                            <TouchableOpacity style={styles.sendButton} onPress={handleSendMessage}>
                                <Text style={styles.sendButtonText}>SEND MESSAGE</Text>
                            </TouchableOpacity>

                            <TouchableOpacity style={styles.keepButton} onPress={onClose}>
                                <Text style={styles.keepButtonText}>Keep Swiping</Text>
                            </TouchableOpacity>

                        </Animated.View>
                    </View>
                )}
            </View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.9)', // Dark overlay
        justifyContent: 'center',
        alignItems: 'center'
    },
    loadingContainer: {
        alignItems: 'center'
    },
    radarRing: {
        marginBottom: 20
    },
    loadingText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '600',
        letterSpacing: 1
    },
    content: {
        width: '100%',
        padding: 20,
        alignItems: 'center'
    },
    successContent: {
        width: '100%',
        alignItems: 'center'
    },
    matchTitle: {
        fontFamily: 'System', // Use custom font if available
        fontSize: 42,
        fontWeight: '900',
        color: '#4ecdc4', // Tinder-ish green/teal mixed or gradient
        fontStyle: 'italic',
        marginBottom: 40,
        textShadowColor: 'rgba(0, 0, 0, 0.5)',
        textShadowOffset: { width: 2, height: 2 },
        textShadowRadius: 5
    },
    avatars: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 30,
        height: 120
    },
    avatarFrame: {
        width: 100,
        height: 100,
        borderRadius: 50,
        borderWidth: 4,
        borderColor: '#fff',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#eee',
        overflow: 'hidden'
    },
    leftAvatar: {
        transform: [{ translateX: 20 }, { rotate: '-10deg' }],
        zIndex: 1
    },
    rightAvatar: {
        transform: [{ translateX: -20 }, { rotate: '10deg' }],
        zIndex: 2
    },
    avatarImage: {
        width: '100%',
        height: '100%'
    },
    placeholderAvatar: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#FF3366',
        width: '100%',
        height: '100%'
    },
    initial: {
        fontSize: 40,
        color: 'white',
        fontWeight: 'bold'
    },
    heartIcon: {
        position: 'absolute',
        backgroundColor: '#FF3366',
        padding: 10,
        borderRadius: 30,
        zIndex: 10,
        bottom: -10,
        elevation: 5
    },
    matchName: {
        color: '#fff',
        fontSize: 18,
        marginBottom: 10,
        textAlign: 'center'
    },
    scoreBadge: {
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        paddingHorizontal: 15,
        paddingVertical: 5,
        borderRadius: 20,
        marginBottom: 20
    },
    scoreText: {
        color: '#4ecdc4',
        fontWeight: 'bold',
        fontSize: 16
    },
    icebreaker: {
        color: '#ccc',
        textAlign: 'center',
        marginBottom: 30,
        fontStyle: 'italic',
        paddingHorizontal: 20
    },
    input: {
        width: '100%',
        backgroundColor: 'rgba(255,255,255,0.1)',
        padding: 15,
        borderRadius: 25,
        color: '#fff',
        marginBottom: 15,
        textAlign: 'center'
    },
    sendButton: {
        backgroundColor: '#FF3366',
        width: '100%',
        padding: 15,
        borderRadius: 25,
        alignItems: 'center',
        marginBottom: 10
    },
    sendButtonText: {
        color: '#fff',
        fontWeight: 'bold',
        fontSize: 16
    },
    keepButton: {
        padding: 15,
    },
    keepButtonText: {
        color: '#ccc',
        fontSize: 16,
        fontWeight: '600'
    },
    tagsContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'center',
        marginBottom: 15
    },
    tagBadge: {
        backgroundColor: 'rgba(255,255,255,0.15)',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 15,
        margin: 4
    },
    tagText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: '600'
    },
    bioText: {
        color: '#eee',
        textAlign: 'center',
        fontSize: 14,
        marginBottom: 20,
        paddingHorizontal: 30,
        fontStyle: 'italic',
        opacity: 0.9
    }
});
