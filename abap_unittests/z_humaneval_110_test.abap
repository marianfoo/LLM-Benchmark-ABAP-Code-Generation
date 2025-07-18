CLASS ltcl_test_exchange DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_exchange IMPLEMENTATION.
  METHOD test_case_1.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    lst2 = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'YES' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    lst2 = VALUE #( ( 1 ) ( 5 ) ( 3 ) ( 4 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ).
    lst2 = VALUE #( ( 2 ) ( 1 ) ( 4 ) ( 3 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'YES' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 5 ) ( 7 ) ( 3 ) ).
    lst2 = VALUE #( ( 2 ) ( 6 ) ( 4 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'YES' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 5 ) ( 7 ) ( 3 ) ).
    lst2 = VALUE #( ( 2 ) ( 6 ) ( 3 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 3 ) ( 2 ) ( 6 ) ( 1 ) ( 8 ) ( 9 ) ).
    lst2 = VALUE #( ( 3 ) ( 5 ) ( 5 ) ( 1 ) ( 1 ) ( 1 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'NO' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lst1   TYPE int4_table,
          lst2   TYPE int4_table,
          result TYPE string.

    lst1 = VALUE #( ( 100 ) ( 200 ) ).
    lst2 = VALUE #( ( 200 ) ( 200 ) ).

    result = z_humaneval_110=>exchange( lst1 = lst1 lst2 = lst2 ).

    cl_abap_unit_assert=>assert_equals( act = result exp = 'YES' ).
  ENDMETHOD.
ENDCLASS.