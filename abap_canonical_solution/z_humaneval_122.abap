CLASS z_humaneval_122 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: add_elements
      IMPORTING
        arr           TYPE int4_table
        k             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_122 IMPLEMENTATION.

  METHOD add_elements.
    DATA: lv_index   TYPE i VALUE 1,
          lv_element TYPE i,
          lv_str     TYPE string,
          lv_len     TYPE i.

    result = 0.

    LOOP AT arr INTO lv_element.
      IF lv_index > k.
        EXIT.
      ENDIF.

      lv_str = |{ lv_element }|.
      lv_len = strlen( lv_str ).

      IF lv_len <= 2.
        result = result + lv_element.
      ENDIF.

      lv_index = lv_index + 1.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.