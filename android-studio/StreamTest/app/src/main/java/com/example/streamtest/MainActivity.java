package com.example.streamtest;

import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;

import com.example.streamtest.R;
import com.example.streamtest.DeveloperKey;
import com.google.android.youtube.player.YouTubeBaseActivity;
import com.google.android.youtube.player.YouTubeInitializationResult;
import com.google.android.youtube.player.YouTubePlayer;
import com.google.android.youtube.player.YouTubePlayerView;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseError;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.security.Provider;


public class MainActivity extends YouTubeBaseActivity implements YouTubePlayer.OnInitializedListener {
    private static final int RECOVERY_DIALOG_REQUEST = 1;

    ProgressDialog mDialog;
    ImageButton btnPlayPause;
    Integer mMaxSpot = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        FirebaseApp.initializeApp(this);
        readMaxSpots();
        updateNumCars();

        //btnPlayPause = (ImageButton)findViewById(R.id.btn_play_pause);
        //btnPlayPause.setOnClickListener(this);

        YouTubePlayerView youTubeView = findViewById(R.id.videoView);
        youTubeView.initialize(DeveloperKey.DEVELOPER_KEY, this);

    }

    @Override
    public void onInitializationSuccess(YouTubePlayer.Provider provider, YouTubePlayer player, boolean wasRestored) {
        if (!wasRestored)
            player.cueVideo("uvzJZ4kp9oY");//"oEgpGv2CF1U");
    }

    @Override
    public void onInitializationFailure(YouTubePlayer.Provider provider,
                                        YouTubeInitializationResult errorReason) {
        if (errorReason.isUserRecoverableError()) {
            errorReason.getErrorDialog(this, RECOVERY_DIALOG_REQUEST).show();
        } else {
            String errorMessage = String.format(getString(R.string.error_player), errorReason.toString());
            Toast.makeText(this, errorMessage, Toast.LENGTH_LONG).show();
        }
    }

    public void readMaxSpots() {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRefMaxSpots = database.getReference("maxSpots");

        myRefMaxSpots.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                mMaxSpot = dataSnapshot.getValue(Integer.class);
                //taskSource.setResult(PresenceType.parse(status));
            }

            @Override
            public void onCancelled(DatabaseError error) {
                //taskSource.setError(firebaseError.toException());
            }
        });

    }
    public void updateNumCars() {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRefCars = database.getReference("cars");
//        DatabaseReference myRefMaxSpots = database.getReference("maxSpots");
//
//        myRefMaxSpots.addListenerForSingleValueEvent(new ValueEventListener() {
//            @Override
//            public void onDataChange(DataSnapshot dataSnapshot) {
//                mMaxSpot = dataSnapshot.getValue(Integer.class);
//                //taskSource.setResult(PresenceType.parse(status));
//            }
//
//            @Override
//            public void onCancelled(DatabaseError error) {
//                //taskSource.setError(firebaseError.toException());
//            }
//        });

        myRefCars.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                // This method is called once with the initial value and again
                // whenever data at this location is updated.
                Integer value = dataSnapshot.getValue(Integer.class);
                //Log.d(LOG_TAG, "Value is: " + value);
                Integer availSpot = 0;
                if (mMaxSpot != null)
                    availSpot = mMaxSpot - value;

                TextView tvNumCars = findViewById(R.id.textViewNumCars);
                tvNumCars.setText(availSpot.toString());
            }

            @Override
            public void onCancelled(DatabaseError error) {
                // Failed to read value
                //Log.w(LOG_TAG, "Failed to read value.", error.toException());
            }
        });
    }

}