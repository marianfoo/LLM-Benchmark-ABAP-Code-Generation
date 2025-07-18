CLASS z_humaneval_109 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS: move_one_ball
      IMPORTING
        VALUE(arr)    TYPE int4_table
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_109 IMPLEMENTATION.

  METHOD move_one_ball.
    DATA: sorted_array TYPE STANDARD TABLE OF i,
          my_arr       TYPE STANDARD TABLE OF i,
          min_value    TYPE i,
          min_index    TYPE sy-tabix,
          temp_arr     TYPE STANDARD TABLE OF i,
          lv_index     TYPE sy-tabix.

    IF lines( arr ) = 0.
      result = abap_true.
      RETURN.
    ENDIF.

    sorted_array = arr.
    SORT sorted_array.

    min_value = arr[ 1 ].
    min_index = 1.
    LOOP AT arr INTO DATA(wa_arr).
      IF wa_arr < min_value.
        min_value = wa_arr.
        min_index = sy-tabix.
      ENDIF.
    ENDLOOP.

    lv_index = min_index.
    WHILE lv_index <= lines( arr ).
      APPEND arr[ lv_index ] TO my_arr.
      lv_index = lv_index + 1.
    ENDWHILE.

    lv_index = 1.
    WHILE lv_index < min_index.
      APPEND arr[ lv_index ] TO my_arr.
      lv_index = lv_index + 1.
    ENDWHILE.

    LOOP AT my_arr INTO DATA(wa_my_arr).
      IF wa_my_arr <> sorted_array[ sy-tabix ].
        result = abap_false.
        RETURN.
      ENDIF.
    ENDLOOP.

    result = abap_true.

  ENDMETHOD.

ENDCLASS.