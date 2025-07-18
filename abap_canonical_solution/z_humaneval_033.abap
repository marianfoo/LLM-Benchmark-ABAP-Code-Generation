CLASS z_humaneval_033 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: sort_third
      IMPORTING
        VALUE(l)      TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_033 IMPLEMENTATION.

  METHOD sort_third.
    DATA: third_elements TYPE int4_table,
          lv_index       TYPE i,
          lv_counter     TYPE i.

    result = l.

    DO lines( result ) TIMES.
      lv_index = sy-index - 1.
      IF lv_index MOD 3 = 0.
        APPEND result[ sy-index ] TO third_elements.
      ENDIF.
    ENDDO.

    SORT third_elements.

    lv_counter = 1.
    DO lines( result ) TIMES.
      lv_index = sy-index - 1.
      IF lv_index MOD 3 = 0.
        result[ sy-index ] = third_elements[ lv_counter ].
        lv_counter = lv_counter + 1.
      ENDIF.
    ENDDO.

  ENDMETHOD.

ENDCLASS.