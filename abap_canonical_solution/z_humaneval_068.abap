CLASS z_humaneval_068 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS: pluck
      IMPORTING
        arr           TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_068 IMPLEMENTATION.

  METHOD pluck.
    DATA: evens     TYPE int4_table,
          min_even  TYPE i,
          min_index TYPE i.

    IF lines( arr ) = 0.
      result = VALUE #( ).
      RETURN.
    ENDIF.

    LOOP AT arr INTO DATA(lv_value).
      IF lv_value MOD 2 = 0.
        evens = VALUE #( BASE evens ( lv_value ) ).
      ENDIF.
    ENDLOOP.

    IF lines( evens ) = 0.
      result = VALUE #( ).
      RETURN.
    ENDIF.

    min_even = evens[ 1 ].
    LOOP AT evens INTO DATA(lv_even).
      IF lv_even < min_even.
        min_even = lv_even.
      ENDIF.
    ENDLOOP.

    LOOP AT arr INTO DATA(lv_arr_value).
      IF lv_arr_value = min_even.
        min_index = sy-tabix - 1.
        EXIT.
      ENDIF.
    ENDLOOP.

    result = VALUE #( ( min_even ) ( min_index ) ).

  ENDMETHOD.

ENDCLASS.