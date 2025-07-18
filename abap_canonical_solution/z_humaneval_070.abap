CLASS z_humaneval_070 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: strange_sort_list
      IMPORTING
        lst           TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_070 IMPLEMENTATION.

  METHOD strange_sort_list.
    DATA: temp_lst TYPE int4_table,
          switch   TYPE abap_bool VALUE abap_true,
          min_val  TYPE i,
          max_val  TYPE i,
          idx      TYPE i.

    temp_lst = lst.

    WHILE lines( temp_lst ) > 0.
      IF switch = abap_true.
        min_val = temp_lst[ 1 ].
        idx = 1.
        LOOP AT temp_lst INTO DATA(val) FROM 2.
          IF val < min_val.
            min_val = val.
            idx = sy-tabix.
          ENDIF.
        ENDLOOP.
        result = VALUE #( BASE result ( min_val ) ).
        DELETE temp_lst INDEX idx.
      ELSE.
        max_val = temp_lst[ 1 ].
        idx = 1.
        LOOP AT temp_lst INTO val FROM 2.
          IF val > max_val.
            max_val = val.
            idx = sy-tabix.
          ENDIF.
        ENDLOOP.
        result = VALUE #( BASE result ( max_val ) ).
        DELETE temp_lst INDEX idx.
      ENDIF.
      switch = xsdbool( switch = abap_false ).
    ENDWHILE.

  ENDMETHOD.

ENDCLASS.