import React from 'react';
import { StyleSheet, Text, View, Image, Dimensions } from 'react-native';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    runOnJS,
} from 'react-native-reanimated';
import FontAwesome from '@expo/vector-icons/FontAwesome';

const { width, height } = Dimensions.get('window');

const SWIPE_THRESHOLD = width * 0.25;

const SocialSwipeCard = ({ item, onSwipeLeft, onSwipeRight }) => {
    const translateX = useSharedValue(0);
    const translateY = useSharedValue(0);

    const panGesture = Gesture.Pan()
        .onUpdate((event) => {
            translateX.value = event.translationX;
            translateY.value = event.translationY;
        })
        .onEnd(() => {
            if (Math.abs(translateX.value) > Math.abs(translateY.value)) {
                if (translateX.value > SWIPE_THRESHOLD) {
                    translateX.value = withSpring(width + 100);
                    runOnJS(onSwipeRight)();
                } else if (translateX.value < -SWIPE_THRESHOLD) {
                    translateX.value = withSpring(-width - 100);
                    runOnJS(onSwipeLeft)();
                } else {
                    translateX.value = withSpring(0);
                    translateY.value = withSpring(0);
                }
            } else {
                translateX.value = withSpring(0);
                translateY.value = withSpring(0);
            }
        });

    const cardStyle = useAnimatedStyle(() => {
        const rotate = `${translateX.value / 20}deg`;
        return {
            transform: [
                { translateX: translateX.value },
                { translateY: translateY.value },
                { rotate: rotate },
            ],
        };
    });

    // Use Robohash for avatars since we don't have profile pics yet
    const avatarUrl = 'https://robohash.org/' + item.username + '?set=set5';

    return (
        <GestureDetector gesture={panGesture}>
            <Animated.View style={[styles.card, cardStyle]}>
                <Image source={{ uri: avatarUrl }} style={styles.image} />

                {/* Match % Badge */}
                <View style={[styles.matchBadge]}>
                    <Text style={styles.matchText}>{(item.similarity * 100).toFixed(0)}% MATCH</Text>
                </View>

                <View style={styles.infoContainer}>
                    <View style={styles.headerRow}>
                        <Text style={styles.username}>{item.username}</Text>
                        <FontAwesome name="check-circle" size={20} color="#0a7ea4" />
                    </View>

                    {item.location_city && (
                        <Text style={styles.location}>📍 {item.location_city}</Text>
                    )}

                    <View style={styles.divider} />

                    {/* ICEBREAKER SECTION */}
                    <Text style={styles.subHeader}>WHY YOU MATCHED</Text>
                    <Text style={styles.icebreaker}>{item.match_reason}</Text>

                    {item.common_movies && item.common_movies.length > 0 && (
                        <View style={styles.commonContainer}>
                            <Text style={styles.commonLabel}>Examples:</Text>
                            <Text style={styles.commonText}>{item.common_movies.join(", ")}</Text>
                        </View>
                    )}

                </View>
            </Animated.View>
        </GestureDetector>
    );
};

const styles = StyleSheet.create({
    card: {
        width: width * 0.9,
        height: height * 0.7,
        backgroundColor: 'white',
        borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
        elevation: 5,
        position: 'absolute',
        alignSelf: 'center',
        top: 50,
    },
    image: {
        width: '100%',
        height: '50%',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        resizeMode: 'cover',
        backgroundColor: '#e1e1e1'
    },
    infoContainer: {
        padding: 20,
    },
    headerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        marginBottom: 5
    },
    username: {
        fontSize: 28,
        fontWeight: 'bold',
        color: '#333',
    },
    location: {
        fontSize: 16,
        color: '#666',
        marginBottom: 15
    },
    divider: {
        height: 1,
        backgroundColor: '#eee',
        marginVertical: 10
    },
    subHeader: {
        fontSize: 12,
        fontWeight: '900',
        color: '#0a7ea4',
        marginBottom: 5,
        letterSpacing: 1
    },
    icebreaker: {
        fontSize: 18,
        fontWeight: '600',
        color: '#333',
        marginBottom: 15,
        fontStyle: 'italic'
    },
    commonContainer: {
        backgroundColor: '#f8f9fa',
        padding: 10,
        borderRadius: 10
    },
    commonLabel: {
        fontSize: 12,
        color: '#888',
        marginBottom: 2
    },
    commonText: {
        fontSize: 14,
        color: '#555'
    },
    matchBadge: {
        position: 'absolute',
        top: 20,
        right: 20,
        backgroundColor: '#FF3366',
        paddingHorizontal: 15,
        paddingVertical: 8,
        borderRadius: 20,
        transform: [{ rotate: '10deg' }],
        elevation: 5,
    },
    matchText: {
        color: 'white',
        fontWeight: '900',
        fontSize: 14
    }
});

export default SocialSwipeCard;
